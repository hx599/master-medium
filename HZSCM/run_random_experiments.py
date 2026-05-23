# -*- coding: utf-8 -*-
"""
随机采样实验脚本。

对每个数据集随机抽取 5 种两两未知类别组合，运行并记录实验结果。
适合快速评估不同 unseen 类别组合对模型性能的影响。

使用方式：
    python run_random_experiments.py --num_repeats 5
    python run_random_experiments.py --num_samples 10 --seed 42
    python run_random_experiments.py --datasets Houston MUUFL
"""

import argparse
import csv
import itertools
import os
import random
import sys
import time

import numpy as np

from config import create_default_args
from data_read_singlemodal import DATASET_CLASS_NAMES
from seg_main import (
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
    parser = argparse.ArgumentParser(description='Random sampling experiment: pick N 2-unseen-class combinations per dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='number of random combinations per dataset (default: 5)')
    parser.add_argument('--num_repeats', type=int, default=5,
                        help='number of repeats per combination (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='random seed for sampling')
    parser.add_argument('--output', type=str, default='Records/singlelabel-single_random_experiments.csv', help='CSV output path')
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                        help='subset of datasets to run, e.g. --datasets Houston MUUFL')
    # 以下参数可覆盖 config 中的默认值
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--pre_epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--fusion_mode', type=str, default='single', choices=['single', 'early'])
    # parser.add_argument('--use_multimodal_pseudo', type=int, default=1,
    #                     help='1: average pseudo-label probabilities from both modalities when available')
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

    # 构建 config overrides
    overrides = {}
    for key in ['epoch', 'pre_epoch', 'lr', 'fusion_mode', 'curr_train_ratio',
                'batch_size', 'beta', 'n_clusters', 'tao', 'threshold']:
        value = getattr(cli_args, key)
        if value is not None:
            overrides[key] = value

    output_path = cli_args.output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 随机采样
    rng = random.Random(cli_args.seed)
    sampled_combos = []
    for ds in datasets:
        nclass = ds['nclass']
        all_combos = list(itertools.combinations(range(nclass), 2))
        n_pick = min(cli_args.num_samples, len(all_combos))
        chosen = rng.sample(all_combos, n_pick)
        sampled_combos.append((ds, chosen))

    total = sum(len(chosen) for _, chosen in sampled_combos)

    print(f"\n{'#'*60}")
    print(f"  Random Sampling Experiment Runner")
    print(f"  Datasets: {[ds['name'] for ds in datasets]}")
    print(f"  Samples per dataset: {cli_args.num_samples}")
    print(f"  Total combinations: {total}")
    print(f"  Repeats per combination: {cli_args.num_repeats}")
    print(f"  Random seed: {cli_args.seed}")
    print(f"  Output: {output_path}")
    print(f"{'#'*60}")

    # 打印采样结果
    for ds, chosen in sampled_combos:
        class_names = DATASET_CLASS_NAMES.get(ds['name'], [f'class_{i}' for i in range(ds['nclass'])])
        print(f"\n  {ds['name']} — {len(chosen)} sampled combinations:")
        for dc in chosen:
            names = [class_names[i] for i in dc]
            print(f"    unseen={list(dc)} ({', '.join(names)})")

    # 加载已完成的组合
    completed = _load_completed(output_path)

    current = 0
    t_start = time.time()

    for ds, chosen in sampled_combos:
        dataset_name = ds['name']
        nclass = ds['nclass']
        class_names = DATASET_CLASS_NAMES.get(dataset_name, [f'class_{i}' for i in range(nclass)])

        print(f"\n>>> Dataset: {dataset_name}")

        for del_class in chosen:
            current += 1
            unseen_key = ';'.join(str(d) for d in del_class)
            combo_key = (dataset_name, unseen_key)
            unseen_names = [class_names[i] for i in del_class]

            if combo_key in completed:
                print(f"  [{current}/{total}] SKIP {dataset_name} unseen={list(del_class)} — already done")
                continue

            print(f"\n  [{current}/{total}] {dataset_name} unseen={list(del_class)} "
                  f"({', '.join(unseen_names)})")

            args = create_default_args(dataset=dataset_name, del_class=list(del_class), **overrides)

            try:
                row = run_experiment_group(args, list(del_class), num_repeats=cli_args.num_repeats)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            save_results_csv([row], output_path, append=True)
            completed.add(combo_key)

            elapsed = time.time() - t_start
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

    _print_final_summary(output_path)


def _print_final_summary(csv_path):
    """按数据集分组打印结果汇总。"""
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
        u_oas = [float(r['U_OA_mean']) for r in rows]

        best_h_oa_idx = np.argmax(h_oas)
        best_h_aa_idx = np.argmax(h_aas)

        print(f"\n  {ds} ({len(rows)} combinations):")
        print(f"    OA:       mean={np.mean(oas):.4f}  best={np.max(oas):.4f}")
        print(f"    U_OA:     mean={np.mean(u_oas):.4f}  best={np.max(u_oas):.4f}")
        print(f"    H_OA:     mean={np.mean(h_oas):.4f}  best={np.max(h_oas):.4f} "
              f"(unseen={rows[best_h_oa_idx]['unseen_names']})")
        print(f"    H_AA:     mean={np.mean(h_aas):.4f}  best={np.max(h_aas):.4f} "
              f"(unseen={rows[best_h_aa_idx]['unseen_names']})")

    # 按数据集打印每个组合的详细结果
    for ds, rows in rows_by_dataset.items():
        print(f"\n  {ds} — all combinations:")
        print(f"    {'unseen':>8s}  {'OA':>8s}  {'U_OA':>8s}  {'H_OA':>8s}  {'H_AA':>8s}")
        for r in sorted(rows, key=lambda x: -float(x['H_OA_mean'])):
            print(f"    {r['unseen_names']:>16s}  {float(r['OA_mean']):8.4f}  "
                  f"{float(r['U_OA_mean']):8.4f}  {float(r['H_OA_mean']):8.4f}  "
                  f"{float(r['H_AA_mean']):8.4f}")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
