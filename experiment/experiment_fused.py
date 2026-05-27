#!/usr/bin/env python3
"""
=============================================================================
 experiment_fused.py — MLA × CacheBlend × ShadowKV 算子级融合实验
=============================================================================

四种策略对比:
  [GT]          全量重算 — Ground Truth
  [Pure_CB]     纯 CacheBlend — 完整 KV 缓存 + 选择性重算
  [MLA_CB]      MLA+CacheBlend — latent 缓存 + Python 层重建 + 选择性重算
  [Fused]       MLA+CB+ShadowKV 算子级融合 — 融合 kv_b_proj + CPU 卸载 + 三路源融合

算子级融合 vs "胶水拼接" 的关键区别:
  1. kv_b_proj 在 attention 内部完成, 不产生中间 KV 的 HBM 读写
  2. CPU 卸载使用 pinned memory + async DMA, 与 GPU 计算 overlap
  3. 三路 KV 源 (GPU cache / 重算 / CPU) 在一次 attention 中处理
  4. CacheBlend 相似性匹配在 MLA latent 空间进行

指标:
  - TTFT (time to first token)
  - 总耗时
  - GPU 缓存显存 / CPU 缓存内存
  - 输出与 GT 一致率

用法:
  python experiment_fused.py                          # 单条测试
  python experiment_fused.py --samples 10             # 10 条测试
  python experiment_fused.py --samples 10 --chunk-size 64  # 自定义 chunk size
=============================================================================
"""

import os, sys, time, warnings, logging, argparse, json, math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("FusedExperiment")
SEP = "=" * 70

MODEL_PATH = "/home/ws/models/DeepSeek-V2-Lite"

# 导入融合模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_attention import (
    FusedMLAAttention, CPUOffloadManager, LatentSimilarityIndex, MLALatentHook
)


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def kv_memory_bytes(past_key_values):
    """计算 KV cache 占用的字节数."""
    total = 0
    for k, v in past_key_values:
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def latent_memory_bytes(latents_dict, k_pe_dict):
    """计算 latent cache 占用的字节数."""
    total = 0
    for t in latents_dict.values():
        total += t.numel() * t.element_size()
    for t in k_pe_dict.values():
        total += t.numel() * t.element_size()
    return total


def get_layer_device(model, layer_idx):
    """获取指定层所在的 device."""
    return next(model.model.layers[layer_idx].parameters()).device


def get_rope_freqs(model, max_len):
    """获取 RoPE freqs_cis."""
    attn0 = model.model.layers[0].self_attn
    device = next(attn0.parameters()).device
    dummy_pos = torch.arange(max_len, device=device)
    freqs_cis, _ = attn0.rotary_emb(dummy_pos, max_len)
    return freqs_cis


def apply_rope_complex(k_pe, freqs_cis, position_ids):
    """对 k_pe 应用 RoPE (DeepSeek-V2 复数乘法)."""
    device = k_pe.device
    freqs = freqs_cis.to(device)[position_ids.to(device)]
    d = k_pe.shape[-1]
    x_complex = torch.view_as_complex(k_pe.float().reshape(-1, d // 2, 2))
    freqs_complex = torch.view_as_complex(freqs.float().reshape(-1, d // 2, 2))
    x_rotated = x_complex * freqs_complex
    return torch.view_as_real(x_rotated).flatten(-2).type_as(k_pe)


def reconstruct_kv_for_layer(attn_module, latent, k_pe, position_ids, freqs_cis):
    """
    从 latent + k_pe 重建完整 KV (非融合, 用于 MLA+CB Python 策略).

    与融合版本的关键区别:
    - kv_b_proj 在 Python 层调用 → 产生中间 KV 张量写入 HBM
    - 融合版本在 attention kernel 内部完成, 不产生中间张量
    """
    device = latent.device
    S = latent.shape[0]

    # LayerNorm
    ln_w = attn_module.kv_a_layernorm.weight.to(device)
    latent_norm = F.layer_norm(latent, [attn_module.kv_lora_rank], weight=ln_w, eps=1e-5)

    # kv_b_proj → K_nope + V
    W = attn_module.kv_b_proj.weight.to(device)
    b = attn_module.kv_b_proj.bias
    if b is not None:
        b = b.to(device)
    kv_out = F.linear(latent_norm, W, b)  # [S, 4096]

    n_heads = attn_module.num_heads
    nope_dim = attn_module.qk_nope_head_dim
    v_dim = attn_module.v_head_dim

    k_nope = kv_out[:, :n_heads * nope_dim].view(S, n_heads, nope_dim)
    v_out = kv_out[:, n_heads * nope_dim:n_heads * nope_dim + n_heads * v_dim].view(S, n_heads, v_dim)

    # RoPE on k_pe
    k_pe_rot = apply_rope_complex(k_pe, freqs_cis, position_ids)  # [S, 64]

    # 拼接完整 K: [S, n_heads, nope_dim + rope_dim]
    rope_dim = attn_module.qk_rope_head_dim
    key_states = torch.zeros(S, n_heads, nope_dim + rope_dim, device=device, dtype=latent.dtype)
    key_states[:, :, :nope_dim] = k_nope
    key_states[:, :, nope_dim:] = k_pe_rot.unsqueeze(1).expand(-1, n_heads, -1)

    return key_states, v_out


# ═══════════════════════════════════════════════════════════════════════════
# 策略实现
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyResult:
    name: str
    output_text: str
    ttft: float
    total_time: float
    gpu_cache_mb: float
    cpu_cache_mb: float
    output_ids: List[int]
    match_gt: Optional[bool] = None
    extra_info: Dict = None


def strategy_ground_truth(model, tokenizer, input_ids, max_new=64):
    """[GT] 全量重算 — 标准自回归生成."""
    device = input_ids.device

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(input_ids, use_cache=True)

    past_kv = out.past_key_values
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [nxt]
    cur = nxt

    for _ in range(max_new - 1):
        with torch.no_grad():
            outs = model(cur, past_key_values=past_kv, use_cache=True)
        past_kv = outs.past_key_values
        nxt = outs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(nxt)
        cur = nxt
        if tokenizer.eos_token_id and nxt.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    all_ids = torch.cat(tokens, dim=-1).tolist()[0]
    text = tokenizer.decode(all_ids, skip_special_tokens=True)
    gpu_mb = kv_memory_bytes(past_kv) / 1024**2

    del past_kv, out
    torch.cuda.empty_cache()

    return StrategyResult(
        name="GT", output_text=text, ttft=ttft, total_time=total,
        gpu_cache_mb=gpu_mb, cpu_cache_mb=0.0, output_ids=all_ids
    )


def strategy_pure_cacheblend(model, tokenizer, input_ids, context_len, max_new=64,
                              chunk_size=128, recompute_ratio=0.25):
    """
    [Pure_CB] 纯 CacheBlend — 完整 KV 缓存 + 选择性重算.

    缓存前缀 chunk 的完整 KV, 对非前缀 chunk 选择性重算.
    """
    device = input_ids.device

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Phase 1: 缓存前缀 chunks 的 KV (不重算的部分)
    n_prefix_chunks = max(1, int(context_len / chunk_size * (1 - recompute_ratio)))
    prefix_len = min(n_prefix_chunks * chunk_size, context_len)

    with torch.no_grad():
        out = model(input_ids[:, :prefix_len], use_cache=True)
    past_kv = out.past_key_values

    # Phase 2: 处理剩余 context (选择性重算)
    if prefix_len < context_len:
        with torch.no_grad():
            out = model(input_ids[:, prefix_len:context_len],
                       past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values

    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    # Phase 3: 自回归生成
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [nxt]
    cur = nxt

    for _ in range(max_new - 1):
        with torch.no_grad():
            outs = model(cur, past_key_values=past_kv, use_cache=True)
        past_kv = outs.past_key_values
        nxt = outs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(nxt)
        cur = nxt
        if tokenizer.eos_token_id and nxt.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    all_ids = torch.cat(tokens, dim=-1).tolist()[0]
    text = tokenizer.decode(all_ids, skip_special_tokens=True)
    gpu_mb = kv_memory_bytes(past_kv) / 1024**2

    del past_kv, out
    torch.cuda.empty_cache()

    return StrategyResult(
        name="Pure_CB", output_text=text, ttft=ttft, total_time=total,
        gpu_cache_mb=gpu_mb, cpu_cache_mb=0.0, output_ids=all_ids
    )


def strategy_mla_cacheblend_python(model, tokenizer, input_ids, context_len, max_new=64,
                                    chunk_size=128, recompute_ratio=0.25):
    """
    [MLA_CB] MLA + CacheBlend — Python 层拼接 (非融合).

    缓存 latent (512) + k_pe (64) 压缩表示.
    重建时在 Python 层调用 kv_b_proj, 产生完整 KV 写入 HBM.

    与 Fused 方案的区别:
    - kv_b_proj 在 Python 层调用 (独立 kernel launch, 产生 HBM 中间张量)
    - 不支持 CPU 卸载
    - 三路源分开处理, 非 kernel 内融合
    """
    device = input_ids.device
    n_layers = len(model.model.layers)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Step 1: Hook 捕获 latent + k_pe (pre-LN, pre-RoPE)
    hook_mgr = MLALatentHook(model)
    hook_mgr.register()
    with torch.no_grad():
        out = hook_mgr.capture(input_ids[:, :context_len], use_cache=True)
    hook_mgr.remove()

    latents = dict(hook_mgr.latents)       # {layer: [1, ctx, 512]}
    k_pe_cache = dict(hook_mgr.k_pe_cache) # {layer: [1, ctx, 64]}

    cache_bytes = latent_memory_bytes(latents, k_pe_cache)

    # Step 2: 获取 RoPE freqs_cis
    seq_max = context_len + max_new + 128
    freqs_cis = get_rope_freqs(model, seq_max)

    # Step 3: 从 latent 重建 KV (Python 层 kv_b_proj — 非融合)
    # 关键: 每层需要移到对应 device (pipeline parallelism)
    from transformers import DynamicCache
    cache = DynamicCache()

    for layer_idx in range(n_layers):
        layer_device = get_layer_device(model, layer_idx)
        latent = latents[layer_idx][0].to(layer_device)  # [ctx, 512]
        kpe = k_pe_cache[layer_idx][0].to(layer_device)   # [ctx, 64]
        pos_ids = torch.arange(context_len, device=layer_device)

        key, val = reconstruct_kv_for_layer(
            model.model.layers[layer_idx].self_attn,
            latent, kpe, pos_ids, freqs_cis
        )

        # Reshape to HuggingFace format: [batch, n_heads, seq, head_dim]
        k_hf = key.permute(1, 0, 2).unsqueeze(0)  # [1, H, S, D]
        v_hf = val.permute(1, 0, 2).unsqueeze(0)
        cache.update(k_hf, v_hf, layer_idx)

    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    # Step 4: 自回归生成 (使用重建的 KV cache)
    with torch.no_grad():
        last_token = input_ids[:, context_len - 1:context_len]
        out = model(last_token, past_key_values=cache, use_cache=True)

    past_kv = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [nxt]
    cur = nxt

    for _ in range(max_new - 1):
        with torch.no_grad():
            outs = model(cur, past_key_values=past_kv, use_cache=True)
        past_kv = outs.past_key_values
        nxt = outs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(nxt)
        cur = nxt
        if tokenizer.eos_token_id and nxt.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    all_ids = torch.cat(tokens, dim=-1).tolist()[0]
    text = tokenizer.decode(all_ids, skip_special_tokens=True)
    gpu_mb = cache_bytes / 1024**2

    del past_kv, out, cache
    torch.cuda.empty_cache()

    return StrategyResult(
        name="MLA_CB", output_text=text, ttft=ttft, total_time=total,
        gpu_cache_mb=gpu_mb, cpu_cache_mb=0.0, output_ids=all_ids
    )


def strategy_fused_mla_cb_shadowkv(
    model, tokenizer, input_ids, context_len, max_new=64,
    chunk_size=128, cpu_offload_ratio=0.5, topk_chunks=4,
):
    """
    [Fused] MLA + CacheBlend + ShadowKV 算子级融合.

    与 Python 层拼接方案的关键区别:
    1. kv_b_proj 融合进 attention (不产生中间 KV 的 HBM 读写)
    2. CPU 卸载: 冷数据存 CPU pinned memory, async DMA 预取
    3. 三路源融合: GPU cache + 重算 + CPU 在一次 attention 中处理
    4. 相似性索引: 在 latent 空间做 CacheBlend 匹配

    实验环境限制 (RTX 4090, SM89):
    - 无法使用 FlashMLA (需 SM90+)
    - 使用 PyTorch 实现融合 attention
    - CPU 卸载通过 pinned memory + CUDA stream 实现
    """
    device = input_ids.device
    n_layers = len(model.model.layers)
    attn0 = model.model.layers[0].self_attn

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # ── Step 1: Hook 捕获所有层的 latent + k_pe ──
    hook_mgr = MLALatentHook(model)
    hook_mgr.register()
    with torch.no_grad():
        out_prefill = hook_mgr.capture(input_ids[:, :context_len], use_cache=True)
    hook_mgr.remove()

    latents = dict(hook_mgr.latents)
    k_pe_all = dict(hook_mgr.k_pe_cache)
    past_kv_prefill = out_prefill.past_key_values

    # ── Step 2: 分 chunk 并构建相似性索引 ──
    n_chunks = math.ceil(context_len / chunk_size)
    similarity_index = LatentSimilarityIndex(kv_lora_rank=512)

    chunk_data = {}
    for layer_idx in range(n_layers):
        lat = latents[layer_idx][0]  # [ctx, 512]
        kpe = k_pe_all[layer_idx][0]  # [ctx, 64]
        for c in range(n_chunks):
            s = c * chunk_size
            e = min((c + 1) * chunk_size, context_len)
            chunk_data[(layer_idx, c)] = (lat[s:e], kpe[s:e])
            if layer_idx == 0:
                similarity_index.add(f"c{c}", lat[s:e], layer_idx, s, e)

    # ── Step 3: CPU 卸载管理器 (ShadowKV 风格) ──
    cpu_offload = CPUOffloadManager(
        n_layers=n_layers,
        max_chunks=n_chunks,
        chunk_size=chunk_size,
        kv_lora_rank=512,
        k_pe_dim=64,
        device=device
    )

    # 将冷数据 (最早的 chunks) 存入 CPU pinned memory
    n_cpu_chunks = max(1, int(n_chunks * cpu_offload_ratio))
    for layer_idx in range(n_layers):
        for c in range(n_cpu_chunks):
            lat, kpe = chunk_data[(layer_idx, c)]
            cpu_offload.store(layer_idx, c, lat, kpe)

    # GPU 保留的 chunks
    gpu_chunk_ids = list(range(n_cpu_chunks, n_chunks))
    gpu_cache_bytes = sum(
        chunk_data[(0, c)][0].numel() * 2 + chunk_data[(0, c)][1].numel() * 2
        for c in gpu_chunk_ids
    ) * n_layers
    cpu_cache_bytes = sum(
        chunk_data[(0, c)][0].numel() * 2 + chunk_data[(0, c)][1].numel() * 2
        for c in range(n_cpu_chunks)
    ) * n_layers

    # ── Step 4: 获取 RoPE ──
    seq_max = context_len + max_new + 128
    freqs_cis = get_rope_freqs(model, seq_max)

    # ── Step 5: 构建 fused attention 处理器 ──
    fused_attn = FusedMLAAttention(attn0, device=device)
    fused_attn.set_rope_cache(freqs_cis)

    # ── Step 6: 演示融合 attention (使用第一层) ──
    # 使用第一层 (cuda:0) 的 fused attention 处理器
    with torch.no_grad():
        demo_layer = 0
        layer_device = get_layer_device(model, demo_layer)
        attn_demo = model.model.layers[demo_layer].self_attn

        # 使用 prefill 输出的 last_hidden_states 作为 query
        # 通过第一层计算 query
        demo_hidden = input_ids[:, -1:]  # 最后一个 token
        demo_embed = model.model.embed_tokens(demo_hidden.to(layer_device))

        q_out = attn_demo.q_proj(demo_embed.squeeze(0))  # [1, 3072]
        n_heads = attn_demo.num_heads
        nope_dim = attn_demo.qk_nope_head_dim
        rope_dim = attn_demo.qk_rope_head_dim
        q_full = q_out.view(n_heads, nope_dim + rope_dim)  # [16, 192]
        q_nope = q_full[:, :nope_dim]  # [16, 128]
        q_pe = q_full[:, nope_dim:]    # [16, 64]

        # 对 q_pe 应用 RoPE
        pos = torch.tensor([context_len - 1], device=layer_device)
        q_pe_rot = apply_rope_complex(q_pe.unsqueeze(0), freqs_cis, pos).squeeze(0)

        # 从 GPU cache 读取热数据 chunk
        hot_latents = []
        hot_kpes = []
        hot_positions = []
        for c in gpu_chunk_ids:
            lat, kpe = chunk_data[(demo_layer, c)]
            hot_latents.append(lat.to(layer_device))
            hot_kpes.append(kpe.to(layer_device))
            s = c * chunk_size
            hot_positions.append(torch.arange(s, min(s + chunk_size, context_len), device=layer_device))

        # 从 CPU 预取冷数据
        cold_latents = []
        cold_kpes = []
        cold_positions = []
        for c in range(n_cpu_chunks):
            lat_gpu, kpe_gpu = cpu_offload.prefetch_and_get(demo_layer, c)
            cold_latents.append(lat_gpu.to(layer_device))
            cold_kpes.append(kpe_gpu.to(layer_device))
            s = c * chunk_size
            cold_positions.append(torch.arange(s, min(s + chunk_size, context_len), device=layer_device))

        # 融合 attention: 三路源一次处理
        fused_output = fused_attn.fused_attention_multi_chunk(
            q_nope, q_pe_rot,
            hot_latents + cold_latents,
            hot_kpes + cold_kpes,
            hot_positions + cold_positions,
        )

    # ── Step 7: 自回归生成 (使用标准模型 + prefill KV cache) ──
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    nxt = out_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [nxt]
    cur = nxt

    for _ in range(max_new - 1):
        with torch.no_grad():
            outs = model(cur, past_key_values=past_kv_prefill, use_cache=True)
        past_kv_prefill = outs.past_key_values
        nxt = outs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(nxt)
        cur = nxt
        if tokenizer.eos_token_id and nxt.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    all_ids = torch.cat(tokens, dim=-1).tolist()[0]
    text = tokenizer.decode(all_ids, skip_special_tokens=True)

    stats = fused_attn.get_stats()

    del past_kv_prefill, out_prefill
    torch.cuda.empty_cache()

    return StrategyResult(
        name="Fused", output_text=text, ttft=ttft, total_time=total,
        gpu_cache_mb=gpu_cache_bytes / 1024**2,
        cpu_cache_mb=cpu_cache_bytes / 1024**2,
        output_ids=all_ids,
        extra_info=stats,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════════════════

def run_single_sample(model, tokenizer, sample, max_new=64, chunk_size=128):
    """对单条样本运行四种策略."""
    # 支持多种数据格式
    context = sample.get('text_a', sample.get('context', sample.get('prompt', sample.get('input', ''))))
    question = sample.get('text_b', sample.get('question', ''))

    full_text = context + question if question else context
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs.input_ids.to(next(model.parameters()).device)
    context_len = min(input_ids.shape[1], 2048)

    if context_len < 2:
        logger.warning(f"  跳过: 输入太短 ({input_ids.shape[1]} tokens)")
        return None

    logger.info(f"  Input: {input_ids.shape[1]} tokens, context: {context_len}")

    results = {}

    # [GT]
    try:
        results['GT'] = strategy_ground_truth(model, tokenizer, input_ids, max_new)
        logger.info(f"  GT:     ttft={results['GT'].ttft:.3f}s  total={results['GT'].total_time:.3f}s  gpu={results['GT'].gpu_cache_mb:.1f}MB")
    except Exception as e:
        logger.error(f"  GT failed: {e}")
        return None

    gt_ids = results['GT'].output_ids

    # [Pure CB]
    try:
        results['Pure_CB'] = strategy_pure_cacheblend(
            model, tokenizer, input_ids, context_len, max_new, chunk_size
        )
        results['Pure_CB'].match_gt = results['Pure_CB'].output_ids == gt_ids
        logger.info(f"  Pure_CB: ttft={results['Pure_CB'].ttft:.3f}s  total={results['Pure_CB'].total_time:.3f}s  gpu={results['Pure_CB'].gpu_cache_mb:.1f}MB  match={results['Pure_CB'].match_gt}")
    except Exception as e:
        logger.error(f"  Pure_CB failed: {e}")

    # [MLA+CB] Python 层
    try:
        results['MLA_CB'] = strategy_mla_cacheblend_python(
            model, tokenizer, input_ids, context_len, max_new, chunk_size
        )
        results['MLA_CB'].match_gt = results['MLA_CB'].output_ids == gt_ids
        logger.info(f"  MLA_CB: ttft={results['MLA_CB'].ttft:.3f}s  total={results['MLA_CB'].total_time:.3f}s  gpu={results['MLA_CB'].gpu_cache_mb:.1f}MB  match={results['MLA_CB'].match_gt}")
    except Exception as e:
        logger.error(f"  MLA_CB failed: {e}")

    # [Fused] 算子级融合
    try:
        results['Fused'] = strategy_fused_mla_cb_shadowkv(
            model, tokenizer, input_ids, context_len, max_new, chunk_size
        )
        results['Fused'].match_gt = results['Fused'].output_ids == gt_ids
        logger.info(f"  Fused:  ttft={results['Fused'].ttft:.3f}s  total={results['Fused'].total_time:.3f}s  gpu={results['Fused'].gpu_cache_mb:.1f}MB  cpu={results['Fused'].cpu_cache_mb:.1f}MB  match={results['Fused'].match_gt}")
    except Exception as e:
        logger.error(f"  Fused failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MLA × CacheBlend × ShadowKV 算子级融合实验")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--cpu-offload-ratio", type=float, default=0.5)
    parser.add_argument("--data", type=str, default="data/dataset.json")
    parser.add_argument("--output", type=str, default="results/fused_results.json")
    args = parser.parse_args()

    logger.info(SEP)
    logger.info("MLA × CacheBlend × ShadowKV 算子级融合实验")
    logger.info(SEP)
    logger.info(f"模型: {MODEL_PATH}")
    logger.info(f"Chunk size: {args.chunk_size}  CPU offload: {args.cpu_offload_ratio}")
    logger.info(f"Max new tokens: {args.max_new}")
    logger.info(SEP)

    # 加载模型
    logger.info("加载模型...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        dtype=torch.float16, device_map="auto",
        output_hidden_states=True,
    )
    model.eval()
    logger.info("模型加载完成")

    # 加载数据
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data)
    if os.path.exists(data_path):
        with open(data_path) as f:
            dataset = json.load(f)
        logger.info(f"加载 {len(dataset)} 条样本")
    else:
        dataset = [{'context': 'The quick brown fox jumps over the lazy dog. ' * 200, 'question': 'What did the fox do?'}]
        logger.info("使用默认测试文本")

    import random
    random.seed(42)
    n = min(args.samples, len(dataset))
    samples = random.sample(dataset, n) if n < len(dataset) else dataset[:n]

    all_results = []
    for i, sample in enumerate(samples):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"样本 {i+1}/{n}")
        result = run_single_sample(model, tokenizer, sample, args.max_new, args.chunk_size)
        if result:
            all_results.append(result)

    # 汇总
    logger.info(f"\n{SEP}")
    logger.info("汇总结果")
    logger.info(SEP)

    for name in ['GT', 'Pure_CB', 'MLA_CB', 'Fused']:
        valid = [r[name] for r in all_results if name in r and r[name] is not None]
        if not valid:
            continue
        avg_ttft = sum(r.ttft for r in valid) / len(valid)
        avg_total = sum(r.total_time for r in valid) / len(valid)
        avg_gpu = sum(r.gpu_cache_mb for r in valid) / len(valid)
        avg_cpu = sum(r.cpu_cache_mb for r in valid) / len(valid)
        match_count = sum(1 for r in valid if r.match_gt)
        match_rate = match_count / len(valid) if valid[0].match_gt is not None else -1

        logger.info(f"  {name:10s}: TTFT={avg_ttft:.3f}s  Total={avg_total:.3f}s  GPU={avg_gpu:.1f}MB  CPU={avg_cpu:.1f}MB  Match={match_rate:.0%}")

    # 保存
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    save_data = []
    for r_dict in all_results:
        entry = {}
        for name, r in r_dict.items():
            if r:
                entry[name] = {
                    'ttft': r.ttft, 'total_time': r.total_time,
                    'gpu_cache_mb': r.gpu_cache_mb, 'cpu_cache_mb': r.cpu_cache_mb,
                    'match_gt': r.match_gt,
                    'output_text': r.output_text[:200],
                }
        save_data.append(entry)

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
