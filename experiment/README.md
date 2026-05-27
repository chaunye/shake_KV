# MLA+CacheBlend 对比实验

## 实验目标
比较 MLA+CacheBlend（latent 空间缓存） vs 单纯 CacheBlend（full KV 空间缓存）在 KV cache 上的性能差异。

## 文件说明

| 文件 | 说明 |
|------|------|
| `compare_v2.py` | **主实验脚本** — MLA+CacheBlend vs 单纯CacheBlend 对比（TTFT、显存、质量） |
| `compare_mla_cb.py` | 第一个版本（有 bug），被 compare_v2.py 取代 |
| `test.py` | 原始 4 部分验证脚本（环境、MLA精度、CacheBlend、性能基准） |
| `validate_mla_cacheblend_flashmla.py` | 原有验证脚本 |
| `CACHEBLEND_FUSION_ANALYSIS.md` | 构想设计文档 |
| `ssh_helper.py` | SSH 连接助手 |
| `ssh_run.py` | SSH 远程执行器（winpty + 密码认证） |
| `fix_remote.py` | 远程服务器补丁脚本（修复 transformers 5.x 兼容性） |
| `check_remote_env.py` | 远程环境检测 |

## 运行方式

在服务器上：
```bash
cd ~/CacheBlend-MLA
source venv/bin/activate
python compare_v2.py
```
