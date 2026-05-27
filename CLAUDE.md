# CacheBlend-MLA 项目

## 项目结构

```
~/CacheBlend-MLA/
├── experiment/              # 对比实验
│   ├── compare_v2.py       # 主实验：MLA+CacheBlend vs 单纯CacheBlend
│   ├── fused_attention.py  # 算子级融合模块：三路KV源 + CPU卸载 + 融合attention
│   ├── experiment_fused.py # 融合实验：GT vs Pure_CB vs MLA_CB vs Fused(4策略对比)
│   ├── run_batch.py        # 批量运行器（stratified sampling, resume, OOM handling）
│   ├── generate_pdf_report.py  # PDF 报告生成器
│   ├── data/               # 数据集
│   │   ├── dataset.json         # 442 中文样本
│   │   └── dataset_en.json      # 2150 英文样本
│   ├── results/
│   │   ├── final_results.json   # 399 条完整结果（Pure + MLA 双策略）
│   │   └── fused_results.json   # 融合实验结果（4策略对比）
│   ├── reports/
│   │   ├── report_400.pdf       # 中文 PDF 实验报告
│   │   └── ...
│   ├── logs/
│   │   └── batch_output_400.log # 26MB 实验日志
│   ├── legacy/              # 旧版弃用脚本
│   ├── docs/
│   │   └── CACHEBLEND_FUSION_ANALYSIS.md  # 融合方案构想文档
│   └── README.md
├── venv/                   # Python 虚拟环境
├── FlashMLA-main/          # FlashMLA 源码（SM90+，此环境不可用）
├── ShadowKV-main/          # ShadowKV 源码（CPU卸载参考实现）
├── vllm-main/              # vLLM 源码
└── LMCache-main-deprecate/ # 旧版 LMCache
```

## 环境

- **模型**: `/home/ws/models/DeepSeek-V2-Lite`（HuggingFace, 16 layers, 27 attn heads, hidden=2048）
- **Python**: 3.13, 虚拟环境 `~/CacheBlend-MLA/venv`
- **GPU**: 8× NVIDIA RTX 4090 (23.5 GB each, SM89)
- **CUDA**: 12.4, **PyTorch**: 2.6.0+cu124, **Transformers**: 4.57.6
- **MLA 参数**: kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
- **激活**: `source ~/CacheBlend-MLA/venv/bin/activate`
- **精度**: float16, device_map="auto"（pipeline parallelism, 每卡 2 层）

## 实验目标

对比四种 KV cache 优化方案（算子级融合）：

| 策略 | 缓存内容 | GPU 显存 | CPU 内存 | TTFT |
|---|---|---|---|---|
| GT (全量重算) | 完整 KV (1088 MB) | 1088 MB | 0 | 0.740s |
| 纯 CacheBlend | 完整 KV + 选择性重算 | 548 MB | 0 | 0.532s |
| MLA+CB (Python层) | latent(512)+k_pe(64), kv_b_proj在Python层 | 61 MB | 0 | 0.371s |
| **MLA+CB+ShadowKV (融合)** | latent+k_pe, kv_b_proj融合进attention, 冷数据CPU卸载 | **30 MB** | 30 MB | 0.412s |

**算子级融合 vs 胶水拼接的关键区别**:
1. kv_b_proj 在 attention kernel 内部完成（不产生中间 KV 的 HBM 读写）
2. CPU 卸载使用 pinned memory + async DMA（ShadowKV 风格）
3. 三路 KV 源 (GPU cache / 重算 / CPU) 在一次 attention 中处理
4. CacheBlend 相似性匹配在 MLA latent 空间进行

指标：TTFT、总耗时、GPU缓存显存、CPU缓存内存、输出与 GT 一致

## 已知兼容性问题与修复

1. **transformers 5.x 移除了 `is_torch_fx_available`** → 在 modeling_deepseek.py 顶部加函数返回 True
2. **`DynamicCache.get_usable_length` 改名** → 改为 `get_seq_length`
3. **`assert attention_mask is not None`** → 改为 if None 生成零 mask
4. **lmcache 0.4.5 无 `cache_engine` 模块** → import 失败时设 engine=None

## 运行方式

```bash
cd ~/CacheBlend-MLA
source venv/bin/activate
cd experiment && python compare_v2.py          # 单条测试 (MLA vs Pure CB)
python run_batch.py --samples 100              # 批量测试
python generate_pdf_report.py                  # 生成 PDF 报告
python experiment_fused.py --samples 5         # 融合实验 (4策略对比)
python experiment_fused.py --samples 10 --chunk-size 64 --cpu-offload-ratio 0.7  # 自定义参数
```

## 已知限制

- **FlashMLA**: 需要 SM90+（H100），RTX 4090 是 SM89，无法使用。实验仅对比缓存策略，不使用 FlashMLA 加速。
- **显存**: 每卡剩余 ~11 GB（模型权重占 ~12 GB），attention score 矩阵 `O(seq_len² × n_heads × dtype)` 约 14.5 GB @ 15K tokens → 需要 `--max-a-len 10000` 控制。
- **88.8% 节省率**: 由 MLA 的 MQA 架构张量形状决定，与输入内容无关（确定性算术结果）。
