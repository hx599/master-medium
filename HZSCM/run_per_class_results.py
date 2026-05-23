# -*- coding: utf-8 -*-
"""
指定数据集和未知类组合，输出包含“总指标 + 每类测试结果”的 CSV。

功能：
1. 复用现有 HZSCM 实验流程运行指定实验
2. 在最终结果上追加每个类别的测试精度（mean/std）
3. 可选：读取已有总结果 CSV，把匹配行作为基础，再补上每类结果

示例：
    python run_per_class_results.py --dataset Houston --del_class 1 9

    python run_per_class_results.py --dataset MUUFL --del_class 4 8 ^
      --num_repeats 3 ^
      --base_csv D:\Master_medium\Records\singlelabel_multimodal_all_experiments.csv ^
      --output Records\per_class_MUUFL_4_8.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from config import create_default_args
from data_read_singlemodal import DATASET_CLASS_NAMES
from seg_main import DATASET_NCLASS, run_single_repeat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one experiment group and save per-class test results.")
    parser.add_argument("--dataset", type=str, required=True, help="Houston, HS-SAR-Berlin, MUUFL")
    parser.add_argument("--del_class", nargs="+", type=int, required=True, help="unseen class indices (0-based)")
    parser.add_argument("--num_repeats", type=int, default=5, help="number of repeated runs")
    parser.add_argument("--output", type=str, default=None, help="output CSV path")
    parser.add_argument(
        "--base_csv",
        type=str,
        default=None,
        help="optional existing summary CSV; if provided, the matching row will be used as the base row",
    )

    # Common experiment overrides
    parser.add_argument("--fusion_mode", type=str, default="early", choices=["single", "early"])
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--pre_epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--curr_train_ratio", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--tao", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser


def _make_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    unseen_str = "_".join(str(x) for x in args.del_class)
    return Path("Records") / f"{args.dataset}_per_class_{unseen_str}.csv"


def _load_base_row(base_csv: str | None, dataset: str, del_class: List[int]) -> Dict[str, str]:
    if not base_csv:
        return {}

    csv_path = Path(base_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Base CSV not found: {csv_path}")

    unseen_key = ";".join(str(x) for x in del_class)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") == dataset and row.get("unseen_idx") == unseen_key:
                return dict(row)
    raise ValueError(f"Cannot find row for dataset={dataset}, unseen_idx={unseen_key} in {csv_path}")


def _aggregate_scalar(results: List[Dict]) -> Dict[str, float]:
    keys_scalar = [
        "pre_oa", "pre_aa", "pre_kappa",
        "oa", "aa", "kappa",
        "s_oa", "u_oa", "h_oa",
        "s_aa", "u_aa", "h_aa",
        "train_time", "test_time",
    ]
    agg: Dict[str, float] = {}
    for key in keys_scalar:
        values = [float(r[key]) for r in results]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))
    return agg


def _aggregate_per_class(results: List[Dict], dataset: str, del_class: List[int]) -> Dict[str, str]:
    nclass = int(results[0]["nclass"])
    class_names = DATASET_CLASS_NAMES.get(dataset, [f"class_{i}" for i in range(nclass)])

    ac_stack = np.stack([np.asarray(r["ac"], dtype=np.float64) for r in results], axis=0)
    pre_ac_stack = np.stack([np.asarray(r["pre_ac"], dtype=np.float64) for r in results], axis=0)

    row: Dict[str, str] = {}
    for idx in range(nclass):
        class_id_1based = idx + 1
        prefix = f"class_{class_id_1based}"
        row[f"{prefix}_name"] = class_names[idx]
        row[f"{prefix}_is_unseen"] = "1" if idx in del_class else "0"
        row[f"{prefix}_pre_acc_mean"] = f"{np.mean(pre_ac_stack[:, idx]):.6f}"
        row[f"{prefix}_pre_acc_std"] = f"{np.std(pre_ac_stack[:, idx]):.6f}"
        row[f"{prefix}_final_acc_mean"] = f"{np.mean(ac_stack[:, idx]):.6f}"
        row[f"{prefix}_final_acc_std"] = f"{np.std(ac_stack[:, idx]):.6f}"
    return row


def _build_summary_row(args: argparse.Namespace, results: List[Dict], base_row: Dict[str, str]) -> Dict[str, str]:
    dataset = args.dataset
    del_class = list(args.del_class)
    nclass = int(results[0]["nclass"])
    class_names = DATASET_CLASS_NAMES.get(dataset, [f"class_{i}" for i in range(nclass)])
    unseen_names = [class_names[i] for i in del_class]

    row = dict(base_row)
    row.update(
        {
            "dataset": dataset,
            "unseen_idx": ";".join(str(x) for x in del_class),
            "unseen_names": ";".join(unseen_names),
            "nclass": str(nclass),
            "num_repeats": str(args.num_repeats),
            "fusion_mode": args.fusion_mode,
        }
    )

    scalar_agg = _aggregate_scalar(results)
    mapped_names = {
        "pre_oa": "pre_OA",
        "pre_aa": "pre_AA",
        "pre_kappa": "pre_Kappa",
        "oa": "OA",
        "aa": "AA",
        "kappa": "Kappa",
        "s_oa": "S_OA",
        "u_oa": "U_OA",
        "h_oa": "H_OA",
        "s_aa": "S_AA",
        "u_aa": "U_AA",
        "h_aa": "H_AA",
        "train_time": "avg_train_time",
        "test_time": "avg_test_time",
    }
    for key, target_name in mapped_names.items():
        row[f"{target_name}_mean"] = f"{scalar_agg[f'{key}_mean']:.6f}"
        row[f"{target_name}_std"] = f"{scalar_agg[f'{key}_std']:.6f}"

    row.update(_aggregate_per_class(results, dataset, del_class))
    return row


def _write_single_row_csv(row: Dict[str, str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _print_per_class_summary(row: Dict[str, str], dataset: str) -> None:
    nclass = int(row["nclass"])
    print("\nPer-class final test accuracy:")
    print("-" * 72)
    print(f"{'class':>5s}  {'type':>7s}  {'name':<30s}  {'final_acc':>16s}")
    print("-" * 72)
    for idx in range(1, nclass + 1):
        prefix = f"class_{idx}"
        kind = "unseen" if row[f"{prefix}_is_unseen"] == "1" else "seen"
        name = row[f"{prefix}_name"]
        acc = f"{float(row[f'{prefix}_final_acc_mean']):.4f} ± {float(row[f'{prefix}_final_acc_std']):.4f}"
        print(f"{idx:5d}  {kind:>7s}  {name:<30.30s}  {acc:>16s}")
    print("-" * 72)


def main() -> None:
    # 统一工作目录到当前脚本所在目录，避免 seg_main/get_map 中的相对路径失效。
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)
    (project_dir / "ResultsImage").mkdir(parents=True, exist_ok=True)
    (project_dir / "Records").mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    cli_args = parser.parse_args()

    overrides = {"fusion_mode": cli_args.fusion_mode}
    for key in ["epoch", "pre_epoch", "lr", "curr_train_ratio", "batch_size", "beta", "n_clusters", "tao", "threshold"]:
        value = getattr(cli_args, key)
        if value is not None:
            overrides[key] = value

    args = create_default_args(dataset=cli_args.dataset, del_class=list(cli_args.del_class), **overrides)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Dataset: {cli_args.dataset}")
    print(f"Unseen classes (0-based): {cli_args.del_class}")
    print(f"Repeats: {cli_args.num_repeats}")
    print(f"Fusion mode: {cli_args.fusion_mode}")

    results: List[Dict] = []
    for repeat_idx in range(cli_args.num_repeats):
        print(f"\n--- Repeat {repeat_idx + 1}/{cli_args.num_repeats} ---")
        metrics = run_single_repeat(args, list(cli_args.del_class), repeat_idx, device)
        results.append(metrics)

    base_row = _load_base_row(cli_args.base_csv, cli_args.dataset, list(cli_args.del_class))
    output_row = _build_summary_row(cli_args, results, base_row)

    output_path = _make_output_path(cli_args)
    _write_single_row_csv(output_row, output_path)
    _print_per_class_summary(output_row, cli_args.dataset)
    print(f"\nSaved per-class result CSV to: {output_path}")


if __name__ == "__main__":
    main()
