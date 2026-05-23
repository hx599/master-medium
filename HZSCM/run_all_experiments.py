# -*- coding: utf-8 -*-
"""
批量遍历实验脚本。

在三个数据集（Houston, HS-SAR-Berlin, MUUFL）上，
遍历所有 C(n,2) 种两两未知类别组合，记录每个组合的实验结果。

特性：
- 增量保存：每完成一个组合立即追加写入 CSV，防止中断丢失数据
- 跳过已完成：读取已有 CSV，跳过已记录的 (dataset, unseen_idx) 组合
- 支持命令行覆盖 epoch、lr 等参数

使用方式：
    python run_all_experiments.py --num_repeats 3 --output Records/all_experiments.csv
    python run_all_experiments.py --num_repeats 1 --epoch 100  # 快速测试
"""

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np

from config import create_default_args
from data_read_singlemodal import DATASET_CLASS_NAMES
from seg_main import (
    CSV_COLUMNS,
    DATASET_NCLASS,
    run_experiment_group,
    save_results_csv,
)


DATASETS = [
    {'name': 'Houston',      'nclass': 15},
    {'name': 'HS-SAR-Berlin', 'nclass': 8},
    {'name': 'MUUFL',        'nclass': 11},
]


def build_parser():
    parser = argparse.ArgumentParser(description='Batch experiment: iterate over all 2-unseen-class combinations')
    parser.add_argument('--num_repeats', type=int, default=5, help='number of repeats per combination')
    parser.add_argument('--output', type=str, default='Records/all_experiments.csv', help='CSV output path')
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                        help='subset of datasets to run, e.g. --datasets Houston MUUFL')
    # 以下参数可覆盖 config 中的默认值
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--pre_epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--fusion_mode', type=str, default=None, choices=['single', 'early'])
    parser.add_argument('--curr_train_ratio', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--n_clusters', type=int, default=None)
    parser.add_argument('--tao', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    return parser


def _load_completed(output_path):
    """读取已有 CSV，返回已完成的 (dataset, unseen_idx) 集合。"""
    completed = set()
    if not os.path.exists(output_path):
        return completed
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['dataset'], row['unseen_idx'])
                completed.add(key)
    except Exception:
        pass
    return completed


def _count_total_combinations(datasets):
    total = 0
    for ds in datasets:
        n = ds['nclass']
        total += len(list(itertools.combinations(range(n), 2)))
    return total


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    # 筛选数据集
    datasets = DATASETS
    if cli_args.datasets is not None:
        selected = set(cli_args.datasets)
        datasets = [ds for ds in datasets if ds['name'] in selected]
        if not datasets:
            print(f'No matching datasets. Available: {[ds["name"] for ds in DATASETS]}')
            sys.exit(1)

    # 构建 config overrides（只覆盖用户显式指定的参数）
    overrides = {}
    for key in ['epoch', 'pre_epoch', 'lr', 'fusion_mode', 'curr_train_ratio',
                'batch_size', 'beta', 'n_clusters', 'tao', 'threshold']:
        value = getattr(cli_args, key)
        if value is not None:
            overrides[key] = value

    output_path = cli_args.output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 加载已完成的组合
    completed = _load_completed(output_path)
    total = _count_total_combinations(datasets)

    print(f"\n{'#'*60}")
    print(f"  Batch Experiment Runner")
    print(f"  Datasets: {[ds['name'] for ds in datasets]}")
    print(f"  Total combinations: {total}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Repeats per combination: {cli_args.num_repeats}")
    print(f"  Output: {output_path}")
    print(f"{'#'*60}\n")

    current = 0
    all_rows = []
    t_start = time.time()

    for ds in datasets:
        dataset_name = ds['name']
        nclass = ds['nclass']
        class_names = DATASET_CLASS_NAMES.get(dataset_name, [f'class_{i}' for i in range(nclass)])
        combinations = list(itertools.combinations(range(nclass), 2))

        print(f"\n>>> Dataset: {dataset_name} ({nclass} classes, {len(combinations)} combinations)")

        for del_class in combinations:
            current += 1
            unseen_key = ';'.join(str(d) for d in del_class)
            combo_key = (dataset_name, unseen_key)

            # 跳过已完成
            if combo_key in completed:
                unseen_names = [class_names[i] for i in del_class]
                print(f"  [{current}/{total}] SKIP {dataset_name} unseen={list(del_class)} "
                      f"({', '.join(unseen_names)}) — already done")
                continue

            unseen_names = [class_names[i] for i in del_class]
            print(f"\n  [{current}/{total}] {dataset_name} unseen={list(del_class)} "
                  f"({', '.join(unseen_names)})")

            # 创建 args
            args = create_default_args(dataset=dataset_name, del_class=list(del_class), **overrides)

            # 运行实验
            try:
                row = run_experiment_group(args, list(del_class), num_repeats=cli_args.num_repeats)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            # 立即追加写入 CSV
            save_results_csv([row], output_path, append=True)
            completed.add(combo_key)
            all_rows.append(row)

            # 进度预估
            elapsed = time.time() - t_start
            done = len(completed) - (len(_load_completed(output_path)) - len(completed))
            if current > 0:
                eta = elapsed / current * (total - current)
                print(f"  Progress: {current}/{total}  Elapsed: {elapsed/60:.1f}min  ETA: {eta/60:.1f}min")

    # 最终汇总
    elapsed = time.time() - t_start
    print(f"\n{'#'*60}")
    print(f"  All experiments completed!")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Results saved to: {output_path}")
    print(f"{'#'*60}")

    # 按数据集打印最佳结果
    _print_final_summary(output_path)


def _print_final_summary(csv_path):
    """按数据集分组打印最佳/平均 H_OA 和 H_AA。"""
    if not os.path.exists(csv_path):
        return

    rows_by_dataset = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row['dataset']
            rows_by_dataset.setdefault(ds, []).append(row)

    if not rows_by_dataset:
        return

    print(f"\n{'='*70}")
    print(f"  Final Summary by Dataset")
    print(f"{'='*70}")

    for ds, rows in rows_by_dataset.items():
        h_oas = [float(r['H_OA_mean']) for r in rows]
        h_aas = [float(r['H_AA_mean']) for r in rows]
        oas = [float(r['OA_mean']) for r in rows]

        best_h_oa_idx = np.argmax(h_oas)
        best_h_aa_idx = np.argmax(h_aas)

        print(f"\n  {ds} ({len(rows)} combinations):")
        print(f"    OA:       mean={np.mean(oas):.4f}  best={np.max(oas):.4f}")
        print(f"    H_OA:     mean={np.mean(h_oas):.4f}  best={np.max(h_oas):.4f} "
              f"(unseen={rows[best_h_oa_idx]['unseen_names']})")
        print(f"    H_AA:     mean={np.mean(h_aas):.4f}  best={np.max(h_aas):.4f} "
              f"(unseen={rows[best_h_aa_idx]['unseen_names']})")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
