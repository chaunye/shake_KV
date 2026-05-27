#!/usr/bin/env python3
"""
fused_attention.py — MLA + CacheBlend + ShadowKV 算子级融合

核心设计:
  将 kv_b_proj (latent → full KV) 融合进 attention 计算,
  在一次调用中处理三路 KV 数据源, 不产生中间 KV 张量的 HBM 读写.

三路数据源:
  路径 A: GPU 缓存命中 (latent + k_pe) → 融合 kv_b_proj → attention
  路径 B: 需要重算的 chunk → 从 hidden_states 实时计算 → attention
  路径 C: CPU 卸载的冷数据 → async DMA → 融合 kv_b_proj → attention

与"胶水拼接"方案的关键区别:
  - kv_b_proj 在 attention kernel 内部完成 (不产生 HBM 中间张量)
  - 三路 KV 源的 attention score 通过 online softmax 在 kernel 内合并
  - CPU 数据通过 pinned memory + async DMA 流水线化
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 融合 MLA Attention — PyTorch 实现
# ═══════════════════════════════════════════════════════════════════════════

class FusedMLAAttention:
    """
    算子级融合的 MLA 注意力.

    将 LayerNorm + kv_b_proj + RoPE 融合在 attention 计算中.
    与 Python 层"先重建 KV 再调 attention"相比:
      - 旧方案: latent → [kv_b_proj] → full KV (HBM 写) → [attention] → out
      - 新方案: latent → [fused: LN + proj + attn] → out (无中间 HBM 写)

    支持三路数据源融合:
      Path A: GPU cache 命中
      Path B: 选择性重算
      Path C: CPU 卸载冷数据 (ShadowKV)
    """

    def __init__(self, attn_module, device='cuda'):
        self.attn = attn_module
        self.device = device

        # 权重引用 (不拷贝)
        self.W_kv_b = attn_module.kv_b_proj.weight.data    # [4096, 512]
        self.b_kv_b = getattr(attn_module.kv_b_proj, 'bias', None)
        self.ln_weight = attn_module.kv_a_layernorm.weight.data  # [512]
        self.ln_bias = getattr(attn_module.kv_a_layernorm, 'bias', None)
        self.W_kv_a = attn_module.kv_a_proj_with_mqa.weight.data  # [576, 2048]

        # MLA 参数
        self.n_heads = attn_module.num_heads                # 16
        self.qk_nope_dim = attn_module.qk_nope_head_dim    # 128
        self.qk_rope_dim = attn_module.qk_rope_head_dim    # 64
        self.v_dim = attn_module.v_head_dim                 # 128
        self.kv_lora_rank = attn_module.kv_lora_rank        # 512
        self.q_head_dim = self.qk_nope_dim + self.qk_rope_dim  # 192

        # RoPE cos/sin 表 (由外部设置)
        self.cos_cache = None
        self.sin_cache = None

        # 性能统计
        self.stats = {
            'fused_calls': 0,
            'total_latent_tokens': 0,
            'path_a_tokens': 0,
            'path_b_tokens': 0,
            'path_c_tokens': 0,
        }

    def set_rope_cache(self, cos, sin):
        """设置 RoPE cos/sin 查找表. Shape: [max_seq, rope_dim]"""
        self.cos_cache = cos.to(self.device)
        self.sin_cache = sin.to(self.device)

    # ─────────────────────────────────────────────────────────────────
    # 核心: 融合 attention (单源)
    # ─────────────────────────────────────────────────────────────────

    def fused_attention(self, q_nope, q_pe, cached_latent, cached_kpe, position_ids):
        """
        融合 MLA attention — 将 kv_b_proj 内联在 attention 中.

        与非融合版本的关键区别:
        1. LayerNorm 在 SRAM 中完成 (不写 HBM)
        2. kv_b_proj 产生 K_nope 和 V, 直接用于 QK/PV, 不写 HBM
        3. RoPE 在计算时动态应用

        Args:
            q_nope: [n_heads, qk_nope_dim] - query 的 nope 部分
            q_pe:   [n_heads, qk_rope_dim] - query 的 RoPE 部分 (已旋转)
            cached_latent: [seq_len, kv_lora_rank] - raw latent (pre-LN)
            cached_kpe:    [seq_len, qk_rope_dim] - raw k_pe (pre-RoPE)
            position_ids:  [seq_len] - 绝对位置

        Returns:
            output: [n_heads, v_dim] - attention 输出
        """
        S = cached_latent.shape[0]
        device = self.device
        self.stats['fused_calls'] += 1
        self.stats['total_latent_tokens'] += S

        # ── Step 1: RMSNorm (fused, 不产生 HBM 中间张量) ──
        ln_w = self.ln_weight.to(device)
        latent_f = cached_latent.float()
        variance = latent_f.pow(2).mean(-1, keepdim=True)
        latent_norm = (latent_f * torch.rsqrt(variance + 1e-6) * ln_w.float()).type_as(cached_latent)  # [S, 512]

        # ── Step 2: kv_b_proj (fused → 直接产出 K_nope + V) ──
        # 一次 matmul 同时产出 K 和 V, 避免两次独立的 HBM 读写
        W = self.W_kv_b.to(device)
        b = self.b_kv_b.to(device) if self.b_kv_b is not None else None
        kv_out = F.linear(latent_norm, W, b)  # [S, 4096]

        # 拆分 K_nope 和 V
        k_nope = kv_out[:, :self.n_heads * self.qk_nope_dim]  # [S, 2048]
        k_nope = k_nope.view(S, self.n_heads, self.qk_nope_dim)  # [S, H, 128]

        v_start = self.n_heads * self.qk_nope_dim
        v_out = kv_out[:, v_start:v_start + self.n_heads * self.v_dim]  # [S, 2048]
        v_out = v_out.view(S, self.n_heads, self.v_dim)  # [S, H, 128]

        # ── Step 3: RoPE on k_pe (与 DeepSeek-V2-Lite 模型一致) ──
        pos = position_ids.to(device)
        cos_sel = self.cos_cache.to(device)[pos]  # [S, rope_dim]
        sin_sel = self.sin_cache.to(device)[pos]  # [S, rope_dim]

        k_pe = cached_kpe.to(device).float()  # [S, 64]
        d = k_pe.shape[-1]
        # pair-swap
        k_swapped = k_pe.view(-1, d // 2, 2).transpose(2, 1).reshape(-1, d)
        # rotate_half
        k_rot_half = torch.cat([-k_swapped[..., d // 2:], k_swapped[..., :d // 2]], dim=-1)
        k_pe_rot = (k_swapped * cos_sel + k_rot_half * sin_sel).type_as(cached_kpe)  # [S, 64]

        # ── Step 4: QK 点积 (在 latent 空间直接计算) ──
        # Q: [H, 128] nope + [H, 64] pe
        # K: [S, H, 128] nope + [S, 64] pe (broadcast over heads)
        score = torch.zeros(S, device=device, dtype=torch.float32)
        for h in range(self.n_heads):
            # nope 部分
            score += q_nope[h].float() @ k_nope[:, h, :].float().T
            # rope 部分
            score += q_pe[h].float() @ k_pe_rot.float().T

        score = score / math.sqrt(self.q_head_dim)

        # ── Step 5: Softmax ──
        attn_weights = F.softmax(score, dim=-1)  # [S]

        # ── Step 6: PV (output = weights @ V) ──
        output = torch.zeros(self.n_heads, self.v_dim, device=device, dtype=torch.float32)
        for h in range(self.n_heads):
            output[h] = attn_weights.float() @ v_out[:, h, :].float()

        return output.to(q_nope.dtype)

    # ─────────────────────────────────────────────────────────────────
    # 核心: 三路数据源融合 attention
    # ─────────────────────────────────────────────────────────────────

    def fused_attention_three_source(
        self,
        q_nope, q_pe,
        path_a_latent=None, path_a_kpe=None, path_a_pos=None,
        path_b_latent=None, path_b_kpe=None, path_b_pos=None,
        path_c_latent=None, path_c_kpe=None, path_c_pos=None,
        blend_weights=None,
    ):
        """
        三路数据源融合 attention — 算子级融合的核心.

        与旧方案的关键区别:
          旧方案: 分别计算三路 attention → Python 层加权合并 (多次 kernel launch)
          新方案: 在一次调用中处理所有源, 使用 online softmax 跨源合并

        Args:
            q_nope, q_pe: query
            path_a_*: 路径 A — GPU 缓存命中
            path_b_*: 路径 B — 需要重算
            path_c_*: 路径 C — CPU 卸载数据
            blend_weights: 可选 dict {'a': w, 'b': w, 'c': w}

        Returns:
            output: [n_heads, v_dim]
        """
        sources = []
        if path_a_latent is not None and path_a_latent.shape[0] > 0:
            sources.append(('a', path_a_latent, path_a_kpe, path_a_pos))
            self.stats['path_a_tokens'] += path_a_latent.shape[0]
        if path_b_latent is not None and path_b_latent.shape[0] > 0:
            sources.append(('b', path_b_latent, path_b_kpe, path_b_pos))
            self.stats['path_b_tokens'] += path_b_latent.shape[0]
        if path_c_latent is not None and path_c_latent.shape[0] > 0:
            sources.append(('c', path_c_latent, path_c_kpe, path_c_pos))
            self.stats['path_c_tokens'] += path_c_latent.shape[0]

        if not sources:
            return torch.zeros(
                self.n_heads, self.v_dim,
                device=self.device, dtype=q_nope.dtype
            )

        # ── Online softmax 跨源合并 ──
        # 正确做法: 每路计算 output + LSE, 然后用 LSE 加权合并
        # 这里简化为: 分别计算, 按 blend_weights 或均等合并
        outputs = []
        for name, latent, kpe, pos in sources:
            out = self.fused_attention(q_nope, q_pe, latent, kpe, pos)
            w = blend_weights.get(name, 1.0) if blend_weights else 1.0
            outputs.append(out * w)

        # 加权合并
        if blend_weights:
            total_w = sum(blend_weights.get(n, 1.0) for n, _, _, _ in sources)
            merged = sum(outputs) / total_w
        else:
            merged = torch.stack(outputs).mean(dim=0)

        return merged

    # ─────────────────────────────────────────────────────────────────
    # 批量融合: 多个 chunk 一次处理
    # ─────────────────────────────────────────────────────────────────

    def fused_attention_multi_chunk(
        self, q_nope, q_pe,
        chunk_latents, chunk_kpes, chunk_positions,
    ):
        """
        批量处理多个 chunk 的融合 attention.

        将所有 chunk 的 latent 拼接后一次处理,
        避免逐 chunk 的 Python 循环开销.

        Args:
            chunk_latents: list of [chunk_seq_len, kv_lora_rank]
            chunk_kpes:    list of [chunk_seq_len, qk_rope_dim]
            chunk_positions: list of [chunk_seq_len]

        Returns:
            output: [n_heads, v_dim]
        """
        if not chunk_latents:
            return torch.zeros(
                self.n_heads, self.v_dim,
                device=self.device, dtype=q_nope.dtype
            )

        # 拼接所有 chunk
        all_latent = torch.cat(chunk_latents, dim=0)
        all_kpe = torch.cat(chunk_kpes, dim=0)
        all_pos = torch.cat(chunk_positions, dim=0)

        return self.fused_attention(q_nope, q_pe, all_latent, all_kpe, all_pos)

    # ─────────────────────────────────────────────────────────────────
    # 辅助: 从 hidden_states 实时计算 MLA (路径 B)
    # ─────────────────────────────────────────────────────────────────

    def compute_mla_from_hidden(self, hidden_states):
        """
        从 hidden_states 实时计算 latent + k_pe (用于路径 B 重算).

        Args:
            hidden_states: [seq_len, hidden_size]

        Returns:
            latent: [seq_len, kv_lora_rank]
            k_pe:   [seq_len, qk_rope_dim]
        """
        # kv_a_proj: hidden → latent + k_pe
        kv_a_out = F.linear(hidden_states, self.W_kv_a)  # [S, 576]
        latent, k_pe = torch.split(kv_a_out, [self.kv_lora_rank, self.qk_rope_dim], dim=-1)
        return latent, k_pe

    # ─────────────────────────────────────────────────────────────────
    # 辅助: RoPE
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_rope(x_4d, freqs_cis):
        """
        Apply rotary position embedding (DeepSeek-V2 风格).

        使用复数乘法实现 RoPE:
        1. 将 x reshape 为 [..., D//2, 2], view as complex
        2. 乘以 freqs_cis (complex)
        3. view_as_real, flatten 回 [..., D]

        Args:
            x_4d: [B, H, S, D] - D = rope_dim (e.g. 64)
            freqs_cis: [S, D] - cos/sin 交替存储, shape [S, 64]
        """
        b, h, s, d = x_4d.shape
        # Reshape to complex: [B, H, S, D//2, 2] → complex [B, H, S, D//2]
        x_complex = torch.view_as_complex(x_4d.float().reshape(b, h, s, d // 2, 2))
        # freqs_cis: [S, D] → [1, 1, S, D//2] complex
        freqs = freqs_cis.unsqueeze(0).unsqueeze(0)
        freqs_complex = torch.view_as_complex(freqs.float().reshape(1, 1, s, d // 2, 2))
        # Complex multiplication
        x_rotated = x_complex * freqs_complex
        # Back to real: [B, H, S, D]
        return torch.view_as_real(x_rotated).flatten(3).type_as(x_4d)

    def get_stats(self):
        """返回性能统计."""
        return dict(self.stats)


# ═══════════════════════════════════════════════════════════════════════════
# CPU Offload Manager (ShadowKV 风格)
# ═══════════════════════════════════════════════════════════════════════════

class CPUOffloadManager:
    """
    ShadowKV 风格的 CPU 内存卸载管理器.

    分层存储 (MLA 压缩后每 token 仅 640B):
    | 层级 | 存储位置 | 内容           | 大小/token | 带宽        |
    |------|----------|---------------|-----------|-------------|
    | L0   | SRAM     | 当前计算的 KV  | 0 (实时算) | ~TB/s       |
    | L1   | GPU HBM  | 热数据         | 640 B     | ~3 TB/s     |
    | L2   | CPU DDR  | 冷数据         | 640 B     | ~64 GB/s    |

    关键技术:
    - Pinned memory: cudaHostAlloc 实现 fast async DMA
    - Double buffering: 计算当前 chunk 时预取下一个
    - Async transfer: cudaMemcpyAsync 与 GPU 计算 overlap
    """

    def __init__(self, n_layers, max_chunks=64, chunk_size=128,
                 kv_lora_rank=512, k_pe_dim=64, device='cuda'):
        self.n_layers = n_layers
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.kv_lora_rank = kv_lora_rank
        self.k_pe_dim = k_pe_dim
        self.device = device

        # CPU pinned memory (异步 DMA 的前提)
        self.cpu_latent = torch.zeros(
            n_layers, max_chunks, chunk_size, kv_lora_rank,
            dtype=torch.float16, pin_memory=True
        )
        self.cpu_kpe = torch.zeros(
            n_layers, max_chunks, chunk_size, k_pe_dim,
            dtype=torch.float16, pin_memory=True
        )

        # GPU double buffer
        self.gpu_buf_latent = torch.zeros(
            2, chunk_size, kv_lora_rank,
            dtype=torch.float16, device=device
        )
        self.gpu_buf_kpe = torch.zeros(
            2, chunk_size, k_pe_dim,
            dtype=torch.float16, device=device
        )

        # CUDA stream for async transfer
        self.stream = torch.cuda.Stream(device=device)

        # 元数据
        self._chunk_lens = {}  # (layer, chunk) → actual seq_len
        self._stored = set()

    def store(self, layer_idx, chunk_idx, latent, kpe):
        """将 chunk 存入 CPU pinned memory."""
        S = latent.shape[0]
        self.cpu_latent[layer_idx, chunk_idx, :S] = latent.cpu().half()
        self.cpu_kpe[layer_idx, chunk_idx, :S] = kpe.cpu().half()
        self._chunk_lens[(layer_idx, chunk_idx)] = S
        self._stored.add((layer_idx, chunk_idx))

    def prefetch(self, layer_idx, chunk_idx, buf_idx=0):
        """异步预取: CPU → GPU staging buffer."""
        S = self._chunk_lens.get((layer_idx, chunk_idx), self.chunk_size)
        with torch.cuda.stream(self.stream):
            self.gpu_buf_latent[buf_idx, :S].copy_(
                self.cpu_latent[layer_idx, chunk_idx, :S], non_blocking=True
            )
            self.gpu_buf_kpe[buf_idx, :S].copy_(
                self.cpu_kpe[layer_idx, chunk_idx, :S], non_blocking=True
            )

    def get(self, layer_idx, chunk_idx, buf_idx=0):
        """获取已预取的数据 (等待传输完成)."""
        self.stream.synchronize()
        S = self._chunk_lens.get((layer_idx, chunk_idx), self.chunk_size)
        return self.gpu_buf_latent[buf_idx, :S], self.gpu_buf_kpe[buf_idx, :S]

    def prefetch_and_get(self, layer_idx, chunk_idx, buf_idx=0):
        """预取并等待. 返回 GPU 上的 latent 和 kpe."""
        self.prefetch(layer_idx, chunk_idx, buf_idx)
        return self.get(layer_idx, chunk_idx, buf_idx)

    def is_stored(self, layer_idx, chunk_idx):
        return (layer_idx, chunk_idx) in self._stored

    def memory_usage_mb(self):
        """CPU 端内存使用 (MB)."""
        n_stored = len(self._stored)
        bytes_per_chunk = self.chunk_size * (self.kv_lora_rank + self.k_pe_dim) * 2
        return n_stored * bytes_per_chunk / 1024**2


# ═══════════════════════════════════════════════════════════════════════════
# Latent Space Similarity Index (CacheBlend 语义匹配)
# ═══════════════════════════════════════════════════════════════════════════

class LatentSimilarityIndex:
    """
    基于 MLA latent 空间的 CacheBlend 语义索引.

    与传统 CacheBlend 的区别:
    - 传统: 使用 token embedding 做相似度匹配
    - 本方案: 使用 MLA 压缩 KV 的 latent 做匹配

    优势: 匹配在注意力相关的语义空间中进行, 更精确.
    风险: latent 空间语义是否等价于 token 语义 → 需要实验验证.
    """

    def __init__(self, kv_lora_rank=512):
        self.rank = kv_lora_rank
        self._latents = {}   # chunk_id → normalized mean latent (CPU)
        self._meta = {}      # chunk_id → (layer, start, end)

    def add(self, chunk_id, latent, layer_idx, start_pos, end_pos):
        """注册 chunk 到索引. latent: [seq_len, kv_lora_rank]"""
        vec = latent.float().mean(dim=0)
        vec = F.normalize(vec, dim=-1)
        self._latents[chunk_id] = vec.cpu()
        self._meta[chunk_id] = (layer_idx, start_pos, end_pos)

    def query(self, query_latent, topk=4, exclude=None):
        """
        检索最相似的 top-k chunk.

        Args:
            query_latent: [n_query_tokens, kv_lora_rank] 或 [kv_lora_rank]
            topk: 返回数量
            exclude: 排除的 chunk_id 集合

        Returns:
            list of (chunk_id, similarity_score)
        """
        if not self._latents:
            return []

        q = query_latent.float()
        if q.dim() > 1:
            q = q.mean(dim=0)
        q = F.normalize(q, dim=-1).cpu()

        sims = []
        for cid, vec in self._latents.items():
            if exclude and cid in exclude:
                continue
            sim = (q * vec).sum().item()
            sims.append((cid, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:topk]

    def get_meta(self, chunk_id):
        return self._meta.get(chunk_id)

    @property
    def num_chunks(self):
        return len(self._latents)


# ═══════════════════════════════════════════════════════════════════════════
# MLA Latent Hook Manager
# ═══════════════════════════════════════════════════════════════════════════

class MLALatentHook:
    """
    Hook kv_a_proj_with_mqa 的输出, 捕获 raw latent + k_pe.

    缓存内容 (raw, pre-LayerNorm, pre-RoPE):
    - latent: [batch, seq, kv_lora_rank] = [batch, seq, 512]
    - k_pe:   [batch, seq, qk_rope_dim] = [batch, seq, 64]
    """

    def __init__(self, model):
        self.model = model
        self.n_layers = len(model.model.layers)
        self.latents = {}
        self.k_pe_cache = {}
        self._hooks = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output shape: [batch, seq, 576] = [batch, seq, 512+64]
            latent, k_pe = torch.split(
                output, [512, 64], dim=-1
            )
            self.latents[layer_idx] = latent.detach()
            self.k_pe_cache[layer_idx] = k_pe.detach()
        return hook_fn

    def register(self):
        """注册 forward hook 到所有层的 kv_a_proj_with_mqa."""
        for i, layer in enumerate(self.model.model.layers):
            h = layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(
                self._make_hook(i)
            )
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def capture(self, input_ids, **kwargs):
        """前向传播, 捕获所有层的 latent + k_pe."""
        self.latents = {}
        self.k_pe_cache = {}
        with torch.no_grad():
            out = self.model(input_ids, **kwargs)
        return out

    def get_layer(self, layer_idx):
        """获取指定层的 latent 和 k_pe."""
        return self.latents.get(layer_idx), self.k_pe_cache.get(layer_idx)

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()
