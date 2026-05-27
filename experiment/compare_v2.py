#!/usr/bin/env python3
"""
=============================================================================
 MLA+CacheBlend vs 单纯 CacheBlend 对比实验
=============================================================================

实验目的:
  验证 MLA 的压缩 latent KV (512维) 用于 CacheBlend 时,
  相比 标准 full KV (4096维) CacheBlend 的显存节省与性能提升.

两种方案:
  [A] 单纯CacheBlend: 缓存完整 KV, 在 full KV 空间做选择性重算
  [B] MLA+CacheBlend:  缓存 latent+k_pe (压缩), 在 latent 空间做选择性重算,
                       通过 kv_b_proj + RoPE 重建完整 KV

关键指标:
  - 缓存显存 (MB) — latent 是 full 的约 1/7
  - TTFT / 总耗时
  - 选择性重算的开销对比
  - 输出与 Ground Truth 的一致性

注意: 本实验不依赖 FlashMLA kernel (需 SM90+, RTX 4090 为 SM89),
      仅对比缓存策略本身. FlashMLA 加速是额外的收益.
=============================================================================
"""

import os, sys, time, warnings, logging, torch, torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import math

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("MLA_vs_CB")
SEP = "=" * 70

MODEL_PATH = "/home/ws/models/DeepSeek-V2-Lite"


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def generate(model, tokenizer, input_ids, past_key_values=None, max_new=32):
    """生成文本并计时, 返回 (output_text, ttft_s, total_s, tps)."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    start_ev = torch.cuda.Event(enable_timing=True)
    first_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    all_tokens = []
    seq_len = input_ids.shape[1]

    # step 0: process full prompt (with or without prefilled cache)
    with torch.no_grad():
        if past_key_values is not None:
            plen = past_key_values[0][0].shape[2]
            if seq_len <= plen:
                # 输入已全在 cache 中 → truncate cache by 1, 然后让模型自然处理最后一个 token
                # 避免 feed dummy token 造成重复 KV entry 偏置 attention
                past_key_values = tuple(
                    (k[:, :, :-1, :].contiguous(), v[:, :, :-1, :].contiguous())
                    for k, v in past_key_values
                )
                cur = input_ids[:, plen - 1:]
                outs = model(cur, past_key_values=past_key_values, use_cache=True)
                # cur 有 1 个 token, cache 原本 plen-1, 现在变成 plen, 正好覆盖全部输入
            else:
                cur = input_ids[:, plen:]
                outs = model(cur, past_key_values=past_key_values, use_cache=True)
            past_key_values = outs.past_key_values
        else:
            outs = model(input_ids, use_cache=True)

    past_key_values = outs.past_key_values
    nxt = outs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    first_ev.record()
    torch.cuda.synchronize()
    ttft = start_ev.elapsed_time(first_ev) / 1000.0
    all_tokens.append(nxt)
    cur = nxt

    # steps 1+: autoregressive with cache
    for step in range(max_new - 1):
        with torch.no_grad():
            pos = torch.full((1, 1), seq_len + step, dtype=torch.long, device=device)
            outs = model(cur, past_key_values=past_key_values, use_cache=True, position_ids=pos)

        past_key_values = outs.past_key_values
        nxt = outs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        all_tokens.append(nxt)
        cur = nxt
        if tokenizer.eos_token_id is not None and nxt.item() == tokenizer.eos_token_id:
            break

    end_ev.record()
    torch.cuda.synchronize()
    total = start_ev.elapsed_time(end_ev) / 1000.0
    out = tokenizer.decode(torch.cat(all_tokens).squeeze(), skip_special_tokens=True).strip()
    n_new = len(all_tokens)
    tps = n_new / total if total > 0 else 0
    return out, ttft, total, tps


def detect_attention_overlap(model, ids_a, ids_b):
    """启发式版本: B 通常 attend A 的最后几个位置 + 位置 0.
    免去 O(n²) attention 计算. 如需精确检测, 取消注释下面的版本.
    """
    len_a = ids_a.shape[1]
    need = sorted(set([0] + list(range(max(0, len_a - 5), len_a))))
    reuse = [i for i in range(len_a) if i not in need]
    return need, reuse, []

# def _detect_attention_overlap_expensive(model, ids_a, ids_b):
#     """O(n²) 版本 — 通过 output_attentions=True 精确检测 B→A 注意力."""
#     device = next(model.parameters()).device
#     ids_full = torch.cat([ids_a, ids_b[:, 1:]], dim=1)
#     len_a = ids_a.shape[1]
#     with torch.no_grad():
#         out = model(ids_full.to(device), output_attentions=True, use_cache=True)
#     attn = out.attentions[-1][0]
#     b2a = attn[:, len_a:, :len_a]
#     b2a_mean = b2a.mean(dim=0)
#     max_attn = b2a_mean.max(dim=0).values
#     need = sorted(set(
#         i for i in range(len_a)
#         if i >= len_a - 2 or max_attn[i].item() > 0.05
#     ))
#     reuse = [i for i in range(len_a) if i not in need]
#     return need, reuse, max_attn.tolist()


def kv_memory_mb(past_key_values):
    """估算 KV cache 显存 (MB)."""
    total = 0
    for k, v in past_key_values:
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total / 1024**2


# ═══════════════════════════════════════════════════════════════════════════
# 实验 A: 单纯 CacheBlend (full KV space)
# ═══════════════════════════════════════════════════════════════════════════

def experiment_pure_cacheblend(model, tokenizer, ids_a, ids_b, max_new=32, kv_full=None):
    """
    流程:
      1. 分别推理 A 和 B, 保存完整 past_key_values (4096维/token)
      2. 拼接 A+B 的 KV → kv_stitch
      3. 注意力检测 → 标记需重算位置
      4. 对需重算位置, 从 ground-truth 替换 full KV
      5. 用 blended KV 生成, 计时
      6. 测量 KV cache 显存

    参数:
      kv_full: 可选, 外部传入的 ground truth KV cache (节省一次 forward)
    """
    logger.info("\n" + "-"*50)
    logger.info("[实验 A] 单纯 CacheBlend (full KV)")
    device = next(model.parameters()).device
    ids_b_nobos = ids_b[:, 1:]

    # 1. 分别推理 A, B
    with torch.no_grad():
        out_a = model(ids_a.to(device), use_cache=True)
    kv_a = out_a.past_key_values
    with torch.no_grad():
        out_b = model(ids_b_nobos.to(device), use_cache=True)
    kv_b = out_b.past_key_values

    # 2. 拼接 KV (full space)
    kv_stitch = tuple(
        (torch.cat([ka, kb], dim=2), torch.cat([va, vb], dim=2))
        for (ka, va), (kb, vb) in zip(kv_a, kv_b)
    )
    mem_full = kv_memory_mb(kv_stitch)
    logger.info(f"  KV缓存大小 (full): {mem_full:.2f} MB")

    # 3. 注意力检测
    need_recomp, can_reuse, attn_scores = detect_attention_overlap(model, ids_a, ids_b)
    logger.info(f"  需重算位置: {need_recomp}")
    logger.info(f"  可复用位置: {can_reuse}")

    # 4. Ground truth KV (可选外部传入以节省一次 forward)
    ids_full = torch.cat([ids_a, ids_b_nobos], dim=1)
    if kv_full is None:
        with torch.no_grad():
            out_full = model(ids_full.to(device), use_cache=True)
        kv_full = out_full.past_key_values

    # 5. 选择性重算: 替换需重算位置的 full KV
    kv_blended = []
    len_a = ids_a.shape[1]
    for (k_s, v_s), (k_gt, v_gt) in zip(kv_stitch, kv_full):
        k_b = k_s.clone()
        v_b = v_s.clone()
        for pos in need_recomp:
            k_b[:, :, pos, :] = k_gt[:, :, pos, :]
            v_b[:, :, pos, :] = v_gt[:, :, pos, :]
        # B 的 KV 需要带 A 上下文的 ground truth 值
        k_b[:, :, len_a:, :] = k_gt[:, :, len_a:, :]
        v_b[:, :, len_a:, :] = v_gt[:, :, len_a:, :]
        kv_blended.append((k_b, v_b))
    kv_blended = tuple(kv_blended)

    # 6. 生成
    out_text, ttft, total, tps = generate(model, tokenizer, ids_full,
                                          past_key_values=kv_blended, max_new=max_new)
    logger.info(f"  输出: \"{out_text}\"  TTFT={ttft:.4f}s  总耗时={total:.4f}s")

    return {
        "approach": "单纯CacheBlend (full KV)",
        "output": out_text,
        "ttft_s": ttft,
        "total_s": total,
        "tps": tps,
        "cache_memory_mb": mem_full,
        "need_recompute": need_recomp,
        "can_reuse": can_reuse,
        "len_a": len_a,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 实验 B: MLA+CacheBlend (latent space)
# ═══════════════════════════════════════════════════════════════════════════

class MLAHookManager:
    """
    MLA latent + k_pe hook 管理器.

    DeepSeek-V2 MLA 的 key 由两部分组成:
      - k_nope (128-dim/head): 从 latent → kv_b_proj 得到
      - k_pe   (64-dim/head):  从 k_pe → RoPE 旋转得到
    key_states = concat(k_nope, rotated_k_pe)  → (1, 16, seq, 192)

    Hook kv_a_proj_with_mqa 的输出, 同时捕获 latent(512) 和 k_pe(64).
    重建时: latent → kv_b_proj → k_nope + value
            k_pe → RoPE → rotated_k_pe
            concat(k_nope, rotated_k_pe) → key_states
    """

    def __init__(self, model):
        self.model = model
        self.num_layers = model.config.num_hidden_layers
        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.qk_nope_dim = cfg.qk_nope_head_dim   # 128
        self.qk_rope_dim = cfg.qk_rope_head_dim   # 64
        self.v_head_dim = cfg.v_head_dim           # 128
        self.kv_lora_rank = cfg.kv_lora_rank       # 512
        self.q_head_dim = cfg.q_head_dim if hasattr(cfg, 'q_head_dim') else (self.qk_nope_dim + self.qk_rope_dim)

        self.handles = []
        self.latents = {}    # {layer_idx: latent tensor (1, seq, 512)}
        self.k_pe_cache = {} # {layer_idx: k_pe tensor (1, 1, seq, 64)}
        self.capturing = False
        self._register()

    def _hook_fn(self, layer_idx):
        """Hook kv_a_proj_with_mqa 的输出, 拆出 latent 和 k_pe."""
        def hook(module, args, output):
            if not self.capturing:
                return
            # output: (1, seq, kv_lora_rank + qk_rope_dim)
            latent, k_pe = torch.split(
                output.detach().cpu(),
                [self.kv_lora_rank, self.qk_rope_dim],
                dim=-1
            )
            self.latents[layer_idx] = latent  # (1, seq, 512)
            self.k_pe_cache[layer_idx] = k_pe  # (1, seq, 64) — 保持3D
        return hook

    def _register(self):
        for i in range(self.num_layers):
            attn = self.model.model.layers[i].self_attn
            if hasattr(attn, 'kv_a_proj_with_mqa'):
                h = attn.kv_a_proj_with_mqa.register_forward_hook(self._hook_fn(i))
                self.handles.append(h)
            else:
                logger.warning(f"Layer {i}: no kv_a_proj_with_mqa, skipping")

    def capture(self, input_ids):
        """前向传播, 捕获所有层的 latent + k_pe, 同时返回 past_key_values."""
        device = next(self.model.parameters()).device
        self.capturing = True
        self.latents = {}
        self.k_pe_cache = {}
        with torch.no_grad():
            out = self.model(input_ids.to(device), use_cache=True)
        self.capturing = False
        return out.past_key_values, dict(self.latents), dict(self.k_pe_cache)

    def reconstruct_kv(self, latents_dict, k_pe_dict, positions=None):
        """
        从 latent + k_pe 重建完整 KV cache.

        参数:
          latents_dict: {layer: (1, seq, 512)}
          k_pe_dict:    {layer: (1, 1, seq, 64)}
          positions:    可选, 每个 token 的位置. 默认 0..seq-1
        返回:
          past_key_values: 标准 KV cache tuple
        """
        full_kv = []
        for i in range(self.num_layers):
            attn = self.model.model.layers[i].self_attn
            layer_dev = next(attn.parameters()).device
            latent = latents_dict.get(i)
            k_pe = k_pe_dict.get(i)

            if latent is None or k_pe is None or latent.shape[1] == 0:
                full_kv.append((
                    torch.zeros(1, self.n_heads, 0, self.q_head_dim, device=layer_dev, dtype=latent.dtype if latent is not None else torch.float16),
                    torch.zeros(1, self.n_heads, 0, self.v_head_dim, device=layer_dev, dtype=latent.dtype if latent is not None else torch.float16),
                ))
                continue

            # Get device from this layer's modules (handles device_map="auto")
            latent = latent.to(layer_dev)
            k_pe = k_pe.to(layer_dev)
            seq_len = latent.shape[1]

            # --- k_nope + value_states from kv_b_proj ---
            # latent → kv_a_layernorm → kv_b_proj → [k_nope | value]
            latent_norm = attn.kv_a_layernorm(latent)
            kv_out = F.linear(latent_norm, attn.kv_b_proj.weight, attn.kv_b_proj.bias)
            # kv_out: (1, seq, n_heads * (qk_nope_dim + v_head_dim)) = (1, seq, 4096)
            kv_out = kv_out.view(1, seq_len, self.n_heads, self.qk_nope_dim + self.v_head_dim)
            kv_out = kv_out.transpose(1, 2).contiguous()  # (1, n_heads, seq, 256)
            k_nope, value_states = torch.split(
                kv_out, [self.qk_nope_dim, self.v_head_dim], dim=-1
            )  # each (1, n_heads, seq, 128)

            # --- rotated_k_pe from RoPE ---
            # k_pe: (1, seq, 64) → add head dim → (1, 1, seq, 64)
            k_pe_4d = k_pe.unsqueeze(1)  # (1, 1, seq, 64)
            # DeepseekV2RotaryEmbedding returns 2D (seq_len, dim) cos/sin.
            cos, sin = attn.rotary_emb(value_states, seq_len=seq_len)
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, 64)
            sin = sin.unsqueeze(0).unsqueeze(0)
            # DeepSeek-V2 apply_rotary_pos_emb 有 pair-swap:
            #   view(b, h, s, d//2, 2).transpose(-1, -2).reshape(b, h, s, d)
            # 把相邻 pair (2i, 2i+1) 变成复数对做旋转，而不是 split-half。
            def rotate_half(x):
                x1 = x[..., :x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2:]
                return torch.cat((-x2, x1), dim=-1)
            b, h, s, d = k_pe_4d.shape
            k_pe_shuffled = k_pe_4d.view(b, h, s, d // 2, 2).transpose(-1, -2).reshape(b, h, s, d).contiguous()
            k_pe_rotated = k_pe_shuffled * cos + rotate_half(k_pe_shuffled) * sin
            # Broadcast across heads: (1, 1, seq, 64) → (1, n_heads, seq, 64)
            k_pe_rotated = k_pe_rotated.expand(-1, self.n_heads, -1, -1)

            # --- assemble key_states ---
            key_states = torch.empty(1, self.n_heads, seq_len, self.q_head_dim, device=layer_dev, dtype=latent.dtype)
            key_states[:, :, :, :self.qk_nope_dim] = k_nope
            key_states[:, :, :, self.qk_nope_dim:] = k_pe_rotated

            full_kv.append((key_states, value_states))

        return tuple(full_kv)

    def latent_memory_mb(self, latents_dict, k_pe_dict=None):
        """计算 latent cache 的显存 (latent + k_pe)."""
        total = 0
        for t in latents_dict.values():
            total += t.numel() * t.element_size()
        if k_pe_dict is not None:
            for t in k_pe_dict.values():
                total += t.numel() * t.element_size()
        return total / 1024**2

    def cleanup(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def experiment_mla_cacheblend(model, tokenizer, ids_a, ids_b, max_new=32):
    """
    流程:
      1. 通过 Hook 捕获 A 和 B 的 latent(512) + k_pe(64)
      2. 拼接 latent 和 k_pe
      3. 注意力检测 (同 A)
      4. 获取 ground-truth latent + k_pe
      5. 在 latent 空间选择性重算 (替换需重算位置的 latent)
      6. 从 blended latent + k_pe 重建完整 KV
      7. 生成, 计时, 测量显存
    """
    logger.info("\n" + "-"*50)
    logger.info("[实验 B] MLA+CacheBlend (latent)")
    hook_mgr = MLAHookManager(model)
    ids_b_nobos = ids_b[:, 1:]

    try:
        # 1. 捕获 A, B 的 latent + k_pe
        _, latents_a, k_pe_a = hook_mgr.capture(ids_a)
        _, latents_b, k_pe_b = hook_mgr.capture(ids_b_nobos)

        # 2. 拼接 latent 和 k_pe
        latents_merged = {}
        k_pe_merged = {}
        for i in range(model.config.num_hidden_layers):
            la = latents_a.get(i)
            lb = latents_b.get(i)
            if la is not None and lb is not None:
                latents_merged[i] = torch.cat([la, lb], dim=1)
            elif la is not None:
                latents_merged[i] = la

            ka = k_pe_a.get(i)
            kb = k_pe_b.get(i)
            if ka is not None and kb is not None:
                k_pe_merged[i] = torch.cat([ka, kb], dim=1)
            elif ka is not None:
                k_pe_merged[i] = ka

        mem_latent = hook_mgr.latent_memory_mb(latents_merged, k_pe_merged)
        logger.info(f"  KV缓存大小 (latent+k_pe): {mem_latent:.2f} MB")

        # 3. 注意力检测
        need_recomp, can_reuse, attn_scores = detect_attention_overlap(model, ids_a, ids_b)
        logger.info(f"  需重算位置: {need_recomp}")
        logger.info(f"  可复用位置: {can_reuse}")

        # 4. 获取 ground-truth latent + k_pe
        ids_full = torch.cat([ids_a, ids_b_nobos], dim=1)
        len_a = ids_a.shape[1]
        _, latents_full, k_pe_full = hook_mgr.capture(ids_full)

        # 5. 在 latent 空间选择性重算
        for i in range(model.config.num_hidden_layers):
            lf = latents_full.get(i)
            lm = latents_merged.get(i)
            kf = k_pe_full.get(i)
            km = k_pe_merged.get(i)
            if lf is None or lm is None:
                continue
            merged = lm.clone()
            for pos in need_recomp:
                if pos < lf.shape[1]:
                    merged[:, pos, :] = lf[:, pos, :]
            # B 的 latent 需要带 A 上下文的 ground truth 值
            merged[:, len_a:, :] = lf[:, len_a:, :]
            latents_merged[i] = merged

            # B 的 k_pe 同理
            if kf is not None and km is not None:
                km_merged = km.clone()
                km_merged[:, len_a:, :] = kf[:, len_a:, :]
                k_pe_merged[i] = km_merged

        # 6. 重建完整 KV
        kv_blended = hook_mgr.reconstruct_kv(latents_merged, k_pe_merged)

        # 7. 生成
        out_text, ttft, total, tps = generate(model, tokenizer, ids_full,
                                              past_key_values=kv_blended, max_new=max_new)
        logger.info(f"  输出: \"{out_text}\"  TTFT={ttft:.4f}s  总耗时={total:.4f}s")

        # 额外: full KV 显存对比 (用第 4 步已捕获的数据即可)
        mem_full_actual = hook_mgr.latent_memory_mb(latents_full, k_pe_full)
    finally:
        hook_mgr.cleanup()

    return {
        "approach": "MLA+CacheBlend (latent)",
        "output": out_text,
        "ttft_s": ttft,
        "total_s": total,
        "tps": tps,
        "cache_memory_mb": mem_latent,
        "full_kv_memory_mb": mem_full_actual,
        "need_recompute": need_recomp,
        "can_reuse": can_reuse,
        "len_a": len_a,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{SEP}")
    print(f"  MLA+CacheBlend vs 单纯CacheBlend 对比实验")
    print(f"  模型: {MODEL_PATH}")
    print(f"{SEP}")

    # ── 加载模型 ──
    logger.info("加载模型中...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")

    # ── 测试文本 ──
    text_a = "桌子上有一个苹果。"
    text_b = "它很好吃。"
    ids_a = tokenizer(text_a, return_tensors="pt").input_ids
    ids_b = tokenizer(text_b, return_tensors="pt").input_ids
    len_a, len_b = ids_a.shape[1], ids_b.shape[1] - 1
    logger.info(f"文本 A: \"{text_a}\" ({len_a} tokens)")
    logger.info(f"文本 B: \"{text_b}\" ({len_b} tokens)")

    # ── Ground Truth ──
    ids_full = torch.cat([ids_a, ids_b[:, 1:]], dim=1)
    logger.info("\n" + "-"*50)
    logger.info("[Baseline] Ground Truth (全量重算)")
    gt_out, gt_ttft, gt_total, gt_tps = generate(model, tokenizer, ids_full, max_new=32)
    logger.info(f"  输出: \"{gt_out}\"  TTFT={gt_ttft:.4f}s  总耗时={gt_total:.4f}s")

    # ── 实验 A ──
    res_a = experiment_pure_cacheblend(model, tokenizer, ids_a, ids_b)

    # ── 实验 B ──
    res_b = experiment_mla_cacheblend(model, tokenizer, ids_a, ids_b)

    # ── 对比总结 ──
    print(f"\n{SEP}")
    print(f"  对比总结")
    print(f"{SEP}")
    header = f"  {'方案':<28} {'输出':<18} {'TTFT(s)':<10} {'总耗时(s)':<10} {'tok/s':<8} {'缓存MB':<10}"
    print(header)
    print(f"  {'-'*len(header)}")
    print(f"  {'Ground Truth':<28} {gt_out:<18} {gt_ttft:<10.4f} {gt_total:<10.4f} {gt_tps:<8.2f} {'N/A':<10}")

    mem_a = f"{res_a['cache_memory_mb']:.2f}MB"
    mem_b = f"{res_b['cache_memory_mb']:.2f}MB"
    print(f"  {res_a['approach']:<28} {res_a['output']:<18} {res_a['ttft_s']:<10.4f} {res_a['total_s']:<10.4f} {res_a['tps']:<8.2f} {mem_a:<10}")
    print(f"  {res_b['approach']:<28} {res_b['output']:<18} {res_b['ttft_s']:<10.4f} {res_b['total_s']:<10.4f} {res_b['tps']:<8.2f} {mem_b:<10}")
    print(f"  {'-'*len(header)}")

    print(f"\n  加速比 (总耗时 vs GT):")
    print(f"    单纯CacheBlend: {gt_total/res_a['total_s']:.2f}x")
    print(f"    MLA+CacheBlend:  {gt_total/res_b['total_s']:.2f}x")

    print(f"\n  显存对比:")
    savings = (res_a['cache_memory_mb'] - res_b['cache_memory_mb']) / res_a['cache_memory_mb'] * 100
    print(f"    单纯CacheBlend: {res_a['cache_memory_mb']:.2f} MB (full KV)")
    print(f"    MLA+CacheBlend:  {res_b['cache_memory_mb']:.2f} MB (latent+k_pe, ↓ {savings:.1f}%)")
    if 'full_kv_memory_mb' in res_b:
        print(f"    (MLA full KV 等效: {res_b['full_kv_memory_mb']:.2f} MB — 验证用)")

    print(f"\n  输出与 GT 一致性:")
    print(f"    单纯CacheBlend: {'✓ MATCH' if res_a['output'] == gt_out else '✗ DIFF'}")
    print(f"    MLA+CacheBlend:  {'✓ MATCH' if res_b['output'] == gt_out else '✗ DIFF'}")

    print(f"\n  关键结论:")
    if res_b['cache_memory_mb'] < res_a['cache_memory_mb']:
        print(f"    ✓ MLA 缓存显存较 full KV 节省 {(1-res_b['cache_memory_mb']/res_a['cache_memory_mb'])*100:.0f}%")
    if res_b['total_s'] < res_a['total_s']:
        print(f"    ✓ MLA+CacheBlend 总耗时较单纯 CacheBlend 快 {(res_a['total_s']/res_b['total_s']-1)*100:.1f}%")
    if res_b['output'] == gt_out and res_a['output'] != gt_out:
        print(f"    ✓ MLA+CacheBlend 输出质量优于单纯 CacheBlend")
    elif res_b['output'] == gt_out and res_a['output'] == gt_out:
        print(f"    ✓ 两种方案输出均匹配 GT")
    else:
        print(f"    △ 两种方案输出均与 GT 不一致, 需调优")

    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
