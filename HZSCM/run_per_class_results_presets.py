# -*- coding: utf-8 -*-
"""
将指定的 9 组未知类组合写死在代码里，直接运行即可。

当前预设组合（按 0-based 类别编号）：
Houston:
    1/9, 7/3, 10/14
MUUFL:
    0/5, 1/9, 3/10
HS-SAR-Berlin:
    4/5, 4/7, 0/5

运行方式：
    直接在 VSCode 里运行本文件
或：
    python run_per_class_results_presets.py

输出内容：
1. 每个组合一份 per-class CSV
2. 一份总汇总 CSV
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch

from config import create_default_args
from run_per_class_results import (
    _build_summary_row,
    _load_base_row,
    _print_per_class_summary,
    _write_single_row_csv,
)
from seg_main import run_single_repeat


# =========================
# 可以直接在这里改实验配置
# =========================
NUM_REPEATS = 5
FUSION_MODE = "early"
BASE_CSV = r"D:\Master_medium\Records\singlelabel_multimodal_all_experiments.csv"
OUTPUT_DIR = Path(r"D:\Master_medium\HZSCM\Records\per_class_presets")

EXPERIMENT_PRESETS: List[Tuple[str, List[int]]] = [
    ("Houston", [1, 9]),
    ("Houston", [7, 3]),
    ("Houston", [10, 14]),
    ("MUUFL", [0, 5]),
    ("MUUFL", [1, 9]),
    ("MUUFL", [3, 10]),
    ("HS-SAR-Berlin", [4, 5]),
    ("HS-SAR-Berlin", [4, 7]),
    ("HS-SAR-Berlin", [0, 5]),
]


def _make_cli_like_args(dataset: str, del_class: List[int]) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
        del_class=list(del_class),
        num_repeats=NUM_REPEATS,
        fusion_mode=FUSION_MODE,
    )


def _make_output_path(dataset: str, del_class: List[int]) -> Path:
    pair_str = "_".join(str(x) for x in del_class)
    return OUTPUT_DIR / f"{dataset}_per_class_{pair_str}.csv"


def _run_one_experiment(dataset: str, del_class: List[int], device: torch.device) -> Dict[str, str]:
    args = create_default_args(dataset=dataset, del_class=list(del_class), fusion_mode=FUSION_MODE)
    cli_like_args = _make_cli_like_args(dataset, del_class)

    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset}")
    print(f"Unseen classes (0-based): {del_class}")
    print(f"Repeats: {NUM_REPEATS}")
    print(f"Fusion mode: {FUSION_MODE}")
    print(f"{'=' * 80}")

    results = []
    for repeat_idx in range(NUM_REPEATS):
        print(f"\n--- Repeat {repeat_idx + 1}/{NUM_REPEATS} ---")
        metrics = run_single_repeat(args, list(del_class), repeat_idx, device)
        results.append(metrics)

    try:
        base_row = _load_base_row(BASE_CSV, dataset, list(del_class))
    except Exception as exc:
        print(f"Base CSV row not found, continue without it: {exc}")
        base_row = {}

    output_row = _build_summary_row(cli_like_args, results, base_row)
    output_path = _make_output_path(dataset, del_class)
    _write_single_row_csv(output_row, output_path)
    _print_per_class_summary(output_row, dataset)
    print(f"Saved per-class result CSV to: {output_path}")
    return output_row


def _write_summary_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    # 统一工作目录到当前脚本所在目录，避免 seg_main/get_map 中的相对路径失效。
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)

    # 这些目录会被现有训练流程中的相对路径直接使用。
    (project_dir / "ResultsImage").mkdir(parents=True, exist_ok=True)
    (project_dir / "Records").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_rows: List[Dict[str, str]] = []
    for dataset, del_class in EXPERIMENT_PRESETS:
        row = _run_one_experiment(dataset, del_class, device)
        all_rows.append(row)

    summary_path = OUTPUT_DIR / "all_presets_per_class_summary.csv"
    _write_summary_csv(all_rows, summary_path)
    print(f"\nSaved summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
