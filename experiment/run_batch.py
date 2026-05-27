#!/usr/bin/env python3
"""
批量运行 MLA+CacheBlend vs 单纯CacheBlend 对比实验.
支持多数据集、按来源分层采样、分来源统计、中间结果保存.

用法:
  # 基本: 用默认中文数据集跑 10 条
  python experiment/run_batch.py --max-samples 10

  # 英文数据集, 按来源均匀采样 (每个 source 最多 3 条)
  python experiment/run_batch.py --dataset experiment/dataset_en.json --max-samples 30 --stratify

  # 同时跑中英文, 只跑短上下文 (500-2000)
  python experiment/run_batch.py --datasets experiment/dataset.json experiment/dataset_en.json --min-a-len 500 --max-a-len 2000 --max-samples 20

  # 长上下文测试 (注意显存)
  python experiment/run_batch.py --dataset experiment/dataset_en.json --max-a-len 8000 --max-samples 10

  # 恢复中断的运行
  python experiment/run_batch.py --resume experiment/results_partial.json --max-samples 50
"""

import os, sys, json, time, argparse, logging, random
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_v2 import (
    generate, detect_attention_overlap, kv_memory_mb,
    experiment_pure_cacheblend, experiment_mla_cacheblend,
    MODEL_PATH, logger, SEP
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
BATCH_LOGGER = logging.getLogger("BatchRun")


def load_datasets(paths: list[str], max_samples: int = None, stratify: bool = False,
                  min_a_len: int = 50, max_a_len: int = 4000, seed: int = 42) -> list:
    """
    加载一个或多个数据集, 返回统一列表, 每个元素含 ids_a, ids_b, sample 元信息.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    all_candidates = []  # list of (ids_a, ids_b, sample_dict)
    source_counts = {}

    for p in paths:
        with open(p) as f:
            data = json.load(f)
        src_label = Path(p).stem  # 用文件名作为 source 标签
        BATCH_LOGGER.info(f"  加载 {p}: {len(data)} 条")

        for s in data:
            ids_a = tokenizer(s["text_a"], return_tensors="pt").input_ids
            ids_b = tokenizer(s["text_b"], return_tensors="pt").input_ids
            la, lb = ids_a.shape[1], ids_b[:, 1:].shape[1]
            if la < min_a_len or la > max_a_len or lb < 3:
                continue
            source = s.get("source", src_label)
            info = {
                "source": source,
                "len_a": la,
                "len_b": lb,
                "dataset_file": src_label,
            }
            all_candidates.append((ids_a, ids_b, info))
            source_counts[source] = source_counts.get(source, 0) + 1

    BATCH_LOGGER.info(f"  候选: {len(all_candidates)} 条 ({dict(source_counts)})")

    if len(all_candidates) == 0:
        BATCH_LOGGER.error("没有符合长度条件的样本!")
        return []

    random.seed(seed)

    if stratify:
        # 按 source 分组, 每组均匀采样
        grouped: dict[str, list] = {}
        for ids_a, ids_b, info in all_candidates:
            src = info["source"]
            grouped.setdefault(src, []).append((ids_a, ids_b, info))

        if max_samples is not None:
            per_source = max(1, max_samples // len(grouped))
        else:
            per_source = None

        selected = []
        for src, items in sorted(grouped.items()):
            random.shuffle(items)
            n = min(len(items), per_source) if per_source else len(items)
            selected.extend(items[:n])
            BATCH_LOGGER.info(f"    {src}: {len(items)} 候选, 取 {n} 条")

        # 如果还没到 max_samples, 从各组的剩余池子补
        if max_samples and len(selected) < max_samples:
            remainder = []
            for items in grouped.values():
                take = per_source if per_source is not None else len(items)
                if take < len(items):
                    remainder.extend(items[take:])
            random.shuffle(remainder)
            needed = max_samples - len(selected)
            selected.extend(remainder[:needed])
            BATCH_LOGGER.info(f"  补充采样: +{needed} 条 (总计 {len(selected)})")

        random.shuffle(selected)
        if max_samples:
            selected = selected[:max_samples]
        return selected
    else:
        random.shuffle(all_candidates)
        if max_samples:
            all_candidates = all_candidates[:max_samples]
        return all_candidates


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--dataset", help="单个数据集路径 (与 --datasets 二选一)")
    parser.add_argument("--datasets", nargs="+", help="多个数据集路径")
    parser.add_argument("--max-samples", type=int, default=20, help="最大样本数")
    parser.add_argument("--min-a-len", type=int, default=50, help="最小 A 长度")
    parser.add_argument("--max-a-len", type=int, default=4000, help="最大 A 长度")
    parser.add_argument("--max-new", type=int, default=16, help="最大生成 token 数")
    parser.add_argument("--stratify", action="store_true", help="按 source 均匀采样")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", help="从中断的 results JSON 恢复")
    parser.add_argument("--output", default=None, help="结果保存路径 (默认自动生成)")
    parser.add_argument("--skip-pure", action="store_true", help="跳过 Pure CacheBlend (只跑 MLA)")
    parser.add_argument("--skip-mla", action="store_true", help="跳过 MLA+CacheBlend (只跑 Pure)")
    args = parser.parse_args()

    # ── 确定数据集路径 ──
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    if args.dataset:
        dataset_paths = [args.dataset]
    elif args.datasets:
        dataset_paths = args.datasets
    else:
        # 默认: 使用中文数据集
        dataset_paths = [os.path.join(experiment_dir, "dataset.json")]

    # 解析相对路径
    dataset_paths = [p if os.path.isabs(p) else os.path.join(os.getcwd(), p) for p in dataset_paths]
    for p in dataset_paths:
        if not os.path.exists(p):
            BATCH_LOGGER.error(f"数据集不存在: {p}")
            sys.exit(1)

    # ── 加载模型 ──
    print(f"\n{SEP}")
    print(f"  MLA+CacheBlend vs 单纯CacheBlend — 批量对比实验")
    print(f"  模型: {MODEL_PATH}")
    print(f"  数据集: {', '.join(os.path.basename(p) for p in dataset_paths)}")
    print(f"  长度范围: {args.min_a_len} - {args.max_a_len} tokens")
    print(f"  最大样本: {args.max_samples}")
    if args.stratify:
        print(f"  采样策略: 分层 (按 source 均匀分配)")
    print(f"{SEP}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    BATCH_LOGGER.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # ── 加载样本 ──
    if args.resume:
        BATCH_LOGGER.info(f"恢复中断运行: {args.resume}")
        with open(args.resume) as f:
            resume_data = json.load(f)
        # 上次已完成的列表 (每个元素是 dict 含 source, len_a)
        done_list = resume_data.get("pure_cb", []) or resume_data.get("mla_cb", [])
        done_keys = set()
        for entry in done_list:
            if isinstance(entry, dict) and entry is not None:
                src = entry.get("source") or entry.get("approach", "")
                la = entry.get("len_a", 0)
                done_keys.add(f"{la}_{src}")
        BATCH_LOGGER.info(f"  上次已完成: {len(done_keys)} 条")
        # 重新筛选, 跳过已完成的
        candidates = load_datasets(dataset_paths, args.max_samples, args.stratify,
                                   args.min_a_len, args.max_a_len, args.seed)
        valid = []
        skipped = 0
        for ids_a, ids_b, info in candidates:
            key = f"{info['len_a']}_{info['source']}"
            if key in done_keys:
                skipped += 1
                continue
            valid.append((ids_a, ids_b, info))
        BATCH_LOGGER.info(f"  跳过 {skipped} 已完成, 剩余 {len(valid)} 条")
    else:
        valid = load_datasets(dataset_paths, args.max_samples, args.stratify,
                              args.min_a_len, args.max_a_len, args.seed)

    n = len(valid)
    BATCH_LOGGER.info(f"有效样本: {n}")
    if n == 0:
        return

    # ── 逐条实验 ──
    results = {"pure_cb": [], "mla_cb": [], "config": vars(args)}
    source_totals = {}  # {source: {'pure_total': [], 'mla_total': [], ...}}

    for idx, (ids_a, ids_b, info) in enumerate(valid):
        ids_b_nobos = ids_b[:, 1:]
        ids_full = torch.cat([ids_a, ids_b_nobos], dim=1)
        len_a, len_b = info["len_a"], info["len_b"]
        src = info["source"]

        print(f"\n  [{idx+1}/{n}] A={len_a} B={len_b} src={src} [{info['dataset_file']}]")

        # Prefill once for kv_full (shared with Pure CacheBlend)
        try:
            with torch.no_grad():
                out_full = model(ids_full.to(next(model.parameters()).device), use_cache=True)
        except torch.cuda.OutOfMemoryError:
            BATCH_LOGGER.error(f"  OOM at prefill (len={len_a}), skipping sample")
            torch.cuda.empty_cache()
            results["pure_cb"].append(None)
            results["mla_cb"].append(None)
            continue
        kv_full = out_full.past_key_values
        kv_full_mb = kv_memory_mb(kv_full)
        print(f"    kv_full: {kv_full_mb:.0f}MB")

        # ── Pure CacheBlend ──
        res_a = None
        if not args.skip_pure:
            try:
                res_a = experiment_pure_cacheblend(model, tokenizer, ids_a, ids_b,
                                                   max_new=args.max_new, kv_full=kv_full)
                res_a["source"] = src
                res_a["len_a"] = len_a
                results["pure_cb"].append(res_a)
            except torch.cuda.OutOfMemoryError:
                BATCH_LOGGER.error(f"  OOM at Pure CacheBlend (len={len_a}), skipping")
                torch.cuda.empty_cache()
                results["pure_cb"].append(None)
            except Exception as e:
                BATCH_LOGGER.error(f"  Pure CB failed: {e}")
                results["pure_cb"].append(None)

        # ── MLA CacheBlend ──
        res_b = None
        if not args.skip_mla:
            try:
                res_b = experiment_mla_cacheblend(model, tokenizer, ids_a, ids_b,
                                                  max_new=args.max_new)
                res_b["source"] = src
                res_b["len_a"] = len_a
                results["mla_cb"].append(res_b)
            except torch.cuda.OutOfMemoryError:
                BATCH_LOGGER.error(f"  OOM at MLA CacheBlend (len={len_a}), skipping")
                torch.cuda.empty_cache()
                results["mla_cb"].append(None)
            except Exception as e:
                BATCH_LOGGER.error(f"  MLA CB failed: {e}")
                results["mla_cb"].append(None)

        # ── 实时打印 ──
        if res_a and res_b:
            savings = (1 - res_b['cache_memory_mb'] / res_a['cache_memory_mb']) * 100
            print(f"    ▶ 缓存: pure={res_a['cache_memory_mb']:.0f}MB  "
                  f"mla={res_b['cache_memory_mb']:.0f}MB  ({savings:.0f}%节省)")
            print(f"    ▶ 耗时: pure={res_a['total_s']:.3f}s  mla={res_b['total_s']:.3f}s  "
                  f"match={'✓✓' if res_a['output']==res_b['output'] else '✗'}")

            # 按 source 聚合
            source_totals.setdefault(src, {
                'count': 0, 'pure_total': [], 'mla_total': [],
                'pure_mem': [], 'mla_mem': [], 'pure_ttft': [], 'mla_ttft': [],
                'match_both': 0, 'count_a': []
            })
            st = source_totals[src]
            st['count'] += 1
            st['pure_total'].append(res_a['total_s'])
            st['mla_total'].append(res_b['total_s'])
            st['pure_mem'].append(res_a['cache_memory_mb'])
            st['mla_mem'].append(res_b['cache_memory_mb'])
            st['pure_ttft'].append(res_a['ttft_s'])
            st['mla_ttft'].append(res_b['ttft_s'])
            st['count_a'].append(len_a)
            if res_a['output'] == res_b['output']:
                st['match_both'] += 1

        # 定期保存中间结果 (每 5 条)
        if (idx + 1) % 5 == 0:
            save_partial(results, args)
    # end for

    # ── 最终保存 ──
    output_path = save_partial(results, args)
    print(f"\n结果已保存: {output_path}")

    # ── 聚合统计 ──
    print_report(source_totals, n)

    print(f"{SEP}\n")


def save_partial(results, args):
    """保存中间/最终结果到 JSON."""
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output:
        output_path = args.output
    else:
        ts = time.strftime("%m%d_%H%M")
        dataset_tag = "custom"
        if hasattr(args, 'dataset') and args.dataset:
            dataset_tag = Path(args.dataset).stem
        output_path = os.path.join(experiment_dir, f"results_{dataset_tag}_{ts}.json")

    # 序列化: 只保留可 JSON 序列化的字段
    serializable = {
        "config": results.get("config", {}),
        "num_samples": len(results.get("pure_cb", [])),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 为每条结果保存关键指标 (去掉 output 文本节省空间)
    for key in ["pure_cb", "mla_cb"]:
        items = results.get(key, [])
        serializable[key] = []
        for r in items:
            if r is None:
                serializable[key].append(None)
            else:
                serializable[key].append({
                    "approach": r.get("approach"),
                    "source": r.get("source", ""),
                    "ttft_s": r.get("ttft_s"),
                    "total_s": r.get("total_s"),
                    "tps": r.get("tps"),
                    "cache_memory_mb": r.get("cache_memory_mb"),
                    "len_a": r.get("len_a"),
                })

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    return output_path


def print_report(source_totals, total_n):
    """打印按 source 分组的统计报告."""
    print(f"\n{SEP}")
    print(f"  聚合统计 ({total_n} samples)")
    print(f"{SEP}")

    # 总体统计
    all_pure_total = []
    all_mla_total = []
    all_pure_mem = []
    all_mla_mem = []
    all_pure_ttft = []
    all_mla_ttft = []
    all_match = 0
    all_count = 0

    for src, st in sorted(source_totals.items()):
        avg_pure_total = sum(st['pure_total']) / st['count']
        avg_mla_total = sum(st['mla_total']) / st['count']
        avg_pure_mem = sum(st['pure_mem']) / st['count']
        avg_mla_mem = sum(st['mla_mem']) / st['count']
        avg_ttft_pure = sum(st['pure_ttft']) / st['count']
        avg_ttft_mla = sum(st['mla_ttft']) / st['count']
        avg_len_a = sum(st['count_a']) / st['count']
        savings = (1 - avg_mla_mem / avg_pure_mem) * 100
        match_rate = st['match_both'] / st['count'] * 100

        all_pure_total.extend(st['pure_total'])
        all_mla_total.extend(st['mla_total'])
        all_pure_mem.extend(st['pure_mem'])
        all_mla_mem.extend(st['mla_mem'])
        all_pure_ttft.extend(st['pure_ttft'])
        all_mla_ttft.extend(st['mla_ttft'])
        all_match += st['match_both']
        all_count += st['count']

        avg_speedup = (avg_pure_total / avg_mla_total - 1) * 100
        sign = "+" if avg_speedup >= 0 else ""
        print(f"\n  [{src}] {st['count']} 条, 平均 A={avg_len_a:.0f}")
        print(f"    缓存: Pure={avg_pure_mem:.0f}MB  MLA={avg_mla_mem:.0f}MB  "
              f"节省 {savings:.1f}%")
        print(f"    TTFT: Pure={avg_ttft_pure*1000:.2f}ms  MLA={avg_ttft_mla*1000:.2f}ms")
        print(f"    耗时: Pure={avg_pure_total:.4f}s  MLA={avg_mla_total:.4f}s  "
              f"({sign}{avg_speedup:.2f}%)")
        print(f"    一致率: {match_rate:.0f}%")

    # 总体合计
    if all_count > 0:
        avg_pure_total = sum(all_pure_total) / all_count
        avg_mla_total = sum(all_mla_total) / all_count
        avg_pure_mem = sum(all_pure_mem) / all_count
        avg_mla_mem = sum(all_mla_mem) / all_count
        avg_ttft_pure = sum(all_pure_ttft) / all_count
        avg_ttft_mla = sum(all_mla_ttft) / all_count
        savings = (1 - avg_mla_mem / avg_pure_mem) * 100
        avg_speedup = (avg_pure_total / avg_mla_total - 1) * 100

        print(f"\n  {'='*50}")
        print(f"  {'总计':>8}: {all_count} 条")
        print(f"  {'='*50}")
        header = f"  {'指标':<30} {'单纯CacheBlend':<18} {'MLA+CacheBlend':<18}"
        print(header)
        print(f"  {'-'*len(header)}")
        print(f"  {'平均 TTFT (ms)':<30} {avg_ttft_pure*1000:<18.2f} {avg_ttft_mla*1000:<18.2f}")
        print(f"  {'平均总耗时 (s)':<30} {avg_pure_total:<18.4f} {avg_mla_total:<18.4f}")
        print(f"  {'平均缓存 (MB)':<30} {avg_pure_mem:<18.1f} {avg_mla_mem:<18.1f}")
        print(f"  {'缓存节省':<30} {'N/A':<18} {savings:<18.1f}%")
        print(f"  {'加速比':<30} {'baseline':<18} {avg_speedup:<18.2f}%")
        print(f"  {'输出一致率':<30} {'N/A':<18} {all_match/all_count*100:<18.1f}%")

        print(f"\n  关键结论:")
        print(f"    1. MLA 缓存显存较 full KV 平均节省 {savings:.0f}%")
        if avg_speedup > 0:
            print(f"    2. MLA+CacheBlend 平均比纯 CacheBlend 快 {avg_speedup:.1f}%")
        elif avg_speedup < 0:
            print(f"    2. MLA+CacheBlend 平均比纯 CacheBlend 慢 {abs(avg_speedup):.2f}% (重建开销)")
        else:
            print(f"    2. MLA+CacheBlend 耗时与纯 CacheBlend 相当")


if __name__ == "__main__":
    main()
