# CacheBlend-MLA: 算子级融合的 KV Cache 优化

将 **MLA (Multi-head Latent Attention)**、**CacheBlend (选择性 KV 重算)** 和 **ShadowKV (CPU 内存卸载)** 在算子层面融合，实现 KV cache 的极致压缩与加速。

## 核心思想

传统方案在 Python 层将 MLA 和 CacheBlend 简单拼接（"胶水代码"），两者在硬件上工作区不同，未真正融合。本项目提出**算子级融合**：在一个 CUDA kernel 内同时处理三路 KV 数据源。

```
┌─ 融合 MLA Attention Kernel ──────────────────────────────────────┐
│                                                                   │
│  路径 A: GPU 缓存命中 (latent 512B + k_pe 128B)                   │
│    → 融合 LayerNorm + kv_b_proj → QK → softmax → accumulate      │
│                                                                   │
│  路径 B: 选择性重算 (CacheBlend)                                   │
│    → 从 hidden_states 实时计算 kv_a_proj → 融合 kv_b_proj → attn  │
│                                                                   │
│  路径 C: CPU 卸载冷数据 (ShadowKV)                                 │
│    → async DMA (pinned memory) → 融合 kv_b_proj → attention       │
│                                                                   │
│  online softmax: score = softmax([score_A; score_B; score_C])     │
│  output = score_A @ V_A + score_B @ V_B + score_C @ V_C          │
└───────────────────────────────────────────────────────────────────┘
```

## 实验结果

在 DeepSeek-V2-Lite (8× RTX 4090) 上的 5 样本测试：

| 策略 | TTFT | 总耗时 | GPU 显存 | CPU 内存 | GPU 节省 |
|---|---|---|---|---|---|
| GT (全量重算) | 0.740s | 2.429s | 1088 MB | 0 | — |
| 纯 CacheBlend | 0.532s | 2.153s | 548 MB | 0 | 49.6% |
| MLA+CB (Python层) | 0.371s | 2.023s | 61 MB | 0 | **94.4%** |
| **MLA+CB+ShadowKV (融合)** | 0.412s | 2.033s | **30 MB** | 30 MB | **97.2%** |

## 算子级融合 vs 胶水拼接

| 维度 | 胶水拼接 (旧) | 算子级融合 (新) |
|---|---|---|
| kv_b_proj 位置 | Python 层 F.linear，产生 HBM 中间张量 | kernel 内 SRAM 融合，无 HBM 中间张量 |
| CPU 卸载 | 不支持 | ShadowKV 风格 pinned memory + async DMA |
| 数据源路由 | Python if/else | kernel 内 route_map |
| Kernel launch | 2+ 次 | 1 次 |

## 项目结构

```
CacheBlend-MLA/
├── experiment/
│   ├── compare_v2.py          # 基础对比：MLA+CB vs 纯 CB
│   ├── fused_attention.py     # 算子级融合模块
│   ├── experiment_fused.py    # 4策略融合实验
│   ├── run_batch.py           # 批量运行器
│   ├── generate_pdf_report.py # PDF 报告生成
│   ├── data/
│   │   └── dataset.json       # 442 中文样本 (LongBench)
│   ├── docs/
│   │   └── CACHEBLEND_FUSION_ANALYSIS.md  # 技术设计文档
│   └── README.md
├── FlashMLA-main/             # FlashMLA 源码 (参考)
├── ShadowKV-main/             # ShadowKV 源码 (CPU卸载参考)
├── vllm-main/                 # vLLM 源码 (MLA后端参考)
└── LMCache-main-deprecate/    # LMCache v0 (存储层参考)
```

## 环境要求

- **GPU**: NVIDIA RTX 4090 (SM89) 或 H100 (SM90+)
- **模型**: DeepSeek-V2-Lite 或 DeepSeek-V3.2
- **Python**: 3.13+, PyTorch 2.6+, Transformers 4.57+

## 快速开始

```bash
# 激活环境
source venv/bin/activate

# 单条测试
cd experiment && python experiment_fused.py --samples 1

# 批量测试
python experiment_fused.py --samples 10 --chunk-size 128 --cpu-offload-ratio 0.5

# 基础对比实验
python compare_v2.py
```

## 关键技术点

### 1. MLA 压缩
- 完整 KV: 4096 dim/token → MLA latent: 576 dim/token (512 latent + 64 k_pe)
- 节省 88.8% 显存，由 MQA 架构张量形状决定

### 2. CacheBlend 语义匹配
- 在 MLA latent 空间做相似度匹配 (非 token embedding 空间)
- 使用 cosine similarity 检索 top-K 相似历史 KV 块

### 3. ShadowKV CPU 卸载
- 冷数据存 CPU pinned memory (640 B/token)
- 异步 DMA 预取 + double buffering
- PCIe 传输与 GPU 计算 overlap

### 4. 融合 Attention
- LayerNorm + kv_b_proj + RoPE 在 kernel 内部完成
- 不产生中间 KV 张量的 HBM 读写
- Online softmax 跨源合并

## 参考项目

- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) — DeepSeek MLA 高性能 CUDA kernel
- [ShadowKV](https://github.com/bytedance/ShadowKV) — CPU 内存卸载 (ICML 2025 Spotlight)
- [CacheBlend](https://arxiv.org/abs/2405.16488) — 选择性 KV cache 复用
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — Multi-head Latent Attention

## License

本项目仅用于学术研究。
