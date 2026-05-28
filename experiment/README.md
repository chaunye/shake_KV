# MLA x CacheBlend x ShadowKV 对比实验

## 实验目标

比较四种 KV cache 优化策略的性能与精度：

| 策略 | 缓存内容 | GPU 显存 | CPU 内存 | 说明 |
|------|----------|----------|----------|------|
| GT | 完整 KV | 1088 MB | 0 | 全量重算 (Ground Truth) |
| Pure_CB | 完整 KV + 选择性重算 | 548 MB | 0 | CacheBlend: prefix/suffix 独立推理 + kv_stitch |
| MLA_CB | latent(512) + k_pe(64) | 61 MB | 0 | MLA 压缩 + Python 层 kv_b_proj 重建 |
| Fused | latent + k_pe | 30 MB | 30 MB | 算子级融合 + CPU 卸载 (ShadowKV) |

## 文件说明

| 文件 | 说明 |
|------|------|
| `experiment_fused.py` | **主实验脚本** — 4 策略对比 (GT / Pure_CB / MLA_CB / Fused) |
| `fused_attention.py` | 算子级融合模块 — 融合 attention + CPU 卸载 + 相似性索引 |
| `compare_v2.py` | 早期对比脚本: MLA+CB vs 纯 CacheBlend |
| `run_batch.py` | 批量运行器 (stratified sampling, resume, OOM handling) |
| `generate_pdf_report.py` | PDF 报告生成器 |
| `data/dataset.json` | 442 条中文样本 (LongBench) |
| `data/dataset_en.json` | 2150 条英文样本 |

## 运行方式

```bash
cd ~/CacheBlend-MLA
source venv/bin/activate
cd experiment

# 单条测试
python experiment_fused.py --samples 1

# 批量测试
python experiment_fused.py --samples 10 --chunk-size 128 --cpu-offload-ratio 0.5

# 自定义参数
python experiment_fused.py --samples 10 --chunk-size 64 --cpu-offload-ratio 0.7

# 基础对比实验
python compare_v2.py

# 批量运行
python run_batch.py --samples 100
```

## 算子级融合 vs 胶水拼接

| 维度 | 胶水拼接 (MLA_CB) | 算子级融合 (Fused) |
|------|-------------------|-------------------|
| kv_b_proj | Python 层 F.linear, 产生 HBM 中间张量 | kernel 内 SRAM 融合, 无 HBM 中间张量 |
| CPU 卸载 | 不支持 | ShadowKV 风格 pinned memory + async DMA |
| 数据源路由 | Python if/else | kernel 内 route_map |
| Kernel launch | 2+ 次 | 1 次 |
| 精度 | float16 级联误差 | float32 计算, float16 存储 |

## 关键技术

### MLA 压缩
- 完整 KV: 4096 dim/token -> MLA latent: 576 dim/token (512 latent + 64 k_pe)
- 节省 88.8% 显存

### CacheBlend 语义匹配
- prefix/suffix 独立推理 -> kv_stitch -> 启发式检测 stale 位置 -> 选择性重算
- MLA_CB 中在 latent 空间做相似度匹配

### ShadowKV CPU 卸载
- 冷数据存 CPU pinned memory (640 B/token)
- 异步 DMA 预取 + double buffering
- PCIe 传输与 GPU 计算 overlap

### 融合 Attention
- LayerNorm + kv_b_proj + RoPE 在 kernel 内部完成
- 不产生中间 KV 张量的 HBM 读写
- 三路数据源: GPU 缓存命中 / 选择性重算 / CPU 卸载冷数据
