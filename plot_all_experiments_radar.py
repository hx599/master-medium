#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 CSV 文件画每个数据集的雷达图，支持有矫正/无矫正对比。

输出内容：
1. 每个数据集一张总体精度雷达图（S_OA / U_OA / H_OA）
2. 每个数据集一张平均精度雷达图（S_AA / U_AA / H_AA）

使用示例：
    # 单文件模式（只画一个 CSV）
    python plot_all_experiments_radar.py --csv Records\\nocorrect_singlelabel.csv

    # 对比模式（有矫正 vs 无矫正，类别组合保持一致）
    python plot_all_experiments_radar.py ^
      --without_csv Records\\nocorrect_singlelabel.csv ^
      --with_csv Records\\singlelabel_multimodal_all_experiments.csv ^
      --output_dir radar_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = ["Houston", "HS-SAR-Berlin", "MUUFL"]
RADAR_STYLE = {
    "seen": {"color": "#2ca02c", "label": "S"},
    "unseen": {"color": "#ff2d2d", "label": "U"},
    "harmonic": {"color": "#1f4cff", "label": "H"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot radar charts for all unseen-class combinations.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="single CSV path (for single-file mode, no comparison)",
    )
    parser.add_argument(
        "--without_csv",
        type=str,
        default=r"D:\Master_medium\Records\nocorrect_singlelabel_multimodal_all_experiments.csv",
        help="CSV path for results WITHOUT correction",
    )
    parser.add_argument(
        "--with_csv",
        type=str,
        default=r"D:\Master_medium\Records\singlelabel_multimodal_all_experiments.csv",
        help="CSV path for results WITH correction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\Master_medium\radar_outputs",
        help="directory to save radar figures",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="optional dataset filter, e.g. --datasets Houston MUUFL",
    )
    parser.add_argument(
        "--houston_topk",
        type=int,
        default=80,
        help="keep only the top-K Houston combinations ranked by combined harmonic score",
    )
    parser.add_argument("--dpi", type=int, default=600, help="output figure dpi")
    return parser


def _dataset_sort_key(name: str) -> Tuple[int, str]:
    if name in DATASET_ORDER:
        return DATASET_ORDER.index(name), name
    return len(DATASET_ORDER), name


def _parse_unseen_idx(value: str) -> Tuple[int, int]:
    parts = str(value).replace(",", ";").split(";")
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 2:
        return (10**9, 10**9)
    return int(parts[0]), int(parts[1])


def _format_unseen_label(value: str) -> str:
    a, b = _parse_unseen_idx(value)
    if a >= 10**9:
        return str(value)
    # CSV 里的 unseen_idx 是 0-based，这里转成论文展示更常用的 1-based 标签。
    return f"{a + 1}/{b + 1}"


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _prepare_dataframe(csv_path: Path, datasets: Sequence[str] | None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = [
        "dataset",
        "unseen_idx",
        "S_OA_mean",
        "U_OA_mean",
        "H_OA_mean",
        "S_AA_mean",
        "U_AA_mean",
        "H_AA_mean",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    if datasets:
        df = df[df["dataset"].isin(datasets)].copy()

    df["sort_key"] = df["unseen_idx"].map(_parse_unseen_idx)
    df["axis_label"] = df["unseen_idx"].map(_format_unseen_label)
    df = df.sort_values(["dataset", "sort_key"]).reset_index(drop=True)
    return df


def _select_topk_pairs(subset: pd.DataFrame, topk: int) -> pd.DataFrame:
    """Keep the top-K pairs ranked by combined harmonic performance."""
    if topk <= 0 or len(subset) <= topk:
        return subset.copy()

    ranked = subset.copy()
    ranked["combined_score"] = ranked["H_OA_mean"].astype(float) + ranked["H_AA_mean"].astype(float)
    ranked = ranked.sort_values(
        ["combined_score", "H_OA_mean", "H_AA_mean"],
        ascending=False,
    ).head(topk)
    ranked = ranked.sort_values("sort_key").reset_index(drop=True)
    return ranked


def _radar_angles(num_axes: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
    return np.concatenate([angles, [angles[0]]])


def _closed(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.concatenate([arr, [arr[0]]])


def _plot_single_dataset_radar(
    dataset: str,
    labels: Sequence[str],
    seen: Sequence[float],
    unseen: Sequence[float],
    harmonic: Sequence[float],
    chart_title: str,
    legend_suffix: str,
    save_path: Path,
    dpi: int,
) -> None:
    num_axes = len(labels)
    angles = _radar_angles(num_axes)

    seen_vals = _closed(np.asarray(seen) * 100.0)
    unseen_vals = _closed(np.asarray(unseen) * 100.0)
    harmonic_vals = _closed(np.asarray(harmonic) * 100.0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=10)
    ax.set_rlabel_position(180)
    ax.grid(True, color="#b7b7b7", linewidth=1.0, alpha=0.9)
    ax.spines["polar"].set_linewidth(1.5)

    ax.plot(angles, seen_vals, color=RADAR_STYLE["seen"]["color"], linewidth=2.0, label=f"S({legend_suffix})")
    ax.fill(angles, seen_vals, color=RADAR_STYLE["seen"]["color"], alpha=0.18)

    ax.plot(angles, unseen_vals, color=RADAR_STYLE["unseen"]["color"], linewidth=2.0, label=f"U({legend_suffix})")
    ax.fill(angles, unseen_vals, color=RADAR_STYLE["unseen"]["color"], alpha=0.15)

    ax.plot(angles, harmonic_vals, color=RADAR_STYLE["harmonic"]["color"], linewidth=2.0, label=f"H({legend_suffix})")
    ax.fill(angles, harmonic_vals, color=RADAR_STYLE["harmonic"]["color"], alpha=0.14)

    ax.set_title(dataset, fontsize=16, pad=25, fontweight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.20, -0.02), frameon=True, fontsize=12)

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _increment_unseen_idx(idx_str: str) -> str:
    """Increment each number in unseen_idx by 1.
    Converts 0-based ('0;1') to 1-based ('1;2')."""
    parts = str(idx_str).replace(",", ";").split(";")
    incremented = [str(int(p.strip()) + 1) for p in parts if p.strip()]
    return ";".join(incremented)


def _get_common_indices(
    df_without: pd.DataFrame,
    df_with: pd.DataFrame,
    dataset: str,
    houston_topk: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get matching unseen_idx pairs for a dataset.
    For Houston: select top-K from df_with (with correction), then filter df_without to match.
    Handles 0-based vs 1-based differences by incrementing without_csv indices."""
    sub_without = df_without[df_without["dataset"] == dataset].copy()
    sub_with = df_with[df_with["dataset"] == dataset].copy()

    # Increment without_csv unseen_idx to match with_csv (0-based -> 1-based)
    sub_without["_match_key"] = sub_without["unseen_idx"].map(_increment_unseen_idx)
    sub_with["_match_key"] = sub_with["unseen_idx"].astype(str)

    if dataset == "Houston":
        # Select top-K from WITH correction as the reference
        sub_with = _select_topk_pairs(sub_with, houston_topk)
        selected_keys = set(sub_with["_match_key"])
        # Filter WITHOUT correction to match
        sub_without = sub_without[sub_without["_match_key"].isin(selected_keys)].copy()
    else:
        # For other datasets, use intersection based on normalized keys
        keys_without = set(sub_without["_match_key"])
        keys_with = set(sub_with["_match_key"])
        common_keys = keys_without & keys_with
        sub_without = sub_without[sub_without["_match_key"].isin(common_keys)].copy()
        sub_with = sub_with[sub_with["_match_key"].isin(common_keys)].copy()

    # Drop temporary match key column
    sub_without = sub_without.drop(columns=["_match_key"]).sort_values("sort_key").reset_index(drop=True)
    sub_with = sub_with.drop(columns=["_match_key"]).sort_values("sort_key").reset_index(drop=True)
    return sub_without, sub_with


def _plot_dataset_radar(
    dataset: str,
    subset: pd.DataFrame,
    suffix: str,
    output_dir: Path,
    dpi: int,
) -> None:
    labels = subset["axis_label"].tolist()

    oa_path = output_dir / f"{_safe_name(dataset)}_总体精度雷达图_{suffix}.png"
    _plot_single_dataset_radar(
        dataset=dataset,
        labels=labels,
        seen=subset["S_OA_mean"].tolist(),
        unseen=subset["U_OA_mean"].tolist(),
        harmonic=subset["H_OA_mean"].tolist(),
        chart_title="总体精度雷达图",
        legend_suffix="oa",
        save_path=oa_path,
        dpi=dpi,
    )

    aa_path = output_dir / f"{_safe_name(dataset)}_平均精度雷达图_{suffix}.png"
    _plot_single_dataset_radar(
        dataset=dataset,
        labels=labels,
        seen=subset["S_AA_mean"].tolist(),
        unseen=subset["U_AA_mean"].tolist(),
        harmonic=subset["H_AA_mean"].tolist(),
        chart_title="平均精度雷达图",
        legend_suffix="aa",
        save_path=aa_path,
        dpi=dpi,
    )


def generate_radar_charts(
    csv_path: Path,
    output_dir: Path,
    datasets: Optional[Sequence[str]],
    houston_topk: int,
    dpi: int,
) -> None:
    df = _prepare_dataframe(csv_path, datasets)
    if df.empty:
        raise ValueError("No data rows found after filtering.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(df["dataset"].unique(), key=_dataset_sort_key):
        subset = df[df["dataset"] == dataset].copy()
        if dataset == "Houston":
            subset = _select_topk_pairs(subset, houston_topk)
            selected_csv = output_dir / f"{_safe_name(dataset)}_top{houston_topk}_selected.csv"
            subset.drop(columns=["sort_key", "axis_label"], errors="ignore").to_csv(selected_csv, index=False, encoding="utf-8-sig")
            print(f"{dataset}: saved top {len(subset)} selected rows to {selected_csv}")

        _plot_dataset_radar(dataset, subset, "single", output_dir, dpi)


def generate_comparison_radar_charts(
    without_csv: Path,
    with_csv: Path,
    output_dir: Path,
    datasets: Optional[Sequence[str]],
    houston_topk: int,
    dpi: int,
) -> None:
    df_without = _prepare_dataframe(without_csv, datasets)
    df_with = _prepare_dataframe(with_csv, datasets)

    if df_without.empty or df_with.empty:
        raise ValueError("One or both CSV files have no data after filtering.")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_datasets = sorted(set(df_without["dataset"]) | set(df_with["dataset"]), key=_dataset_sort_key)

    for dataset in all_datasets:
        has_without = dataset in df_without["dataset"].values
        has_with = dataset in df_with["dataset"].values

        if not has_without or not has_with:
            print(f"Warning: {dataset} missing in {'without' if not has_without else 'with'} CSV, skipping.")
            continue

        sub_without, sub_with = _get_common_indices(df_without, df_with, dataset, houston_topk)

        if sub_without.empty or sub_with.empty:
            print(f"Warning: No common unseen_idx for {dataset}, skipping.")
            continue

        print(f"{dataset}: {len(sub_without)} common combinations")

        # Save selected indices
        selected_csv = output_dir / f"{_safe_name(dataset)}_common_indices.csv"
        sub_without[["unseen_idx"]].to_csv(selected_csv, index=False, encoding="utf-8-sig")

        _plot_dataset_radar(dataset, sub_without, "无矫正", output_dir, dpi)
        _plot_dataset_radar(dataset, sub_with, "有矫正", output_dir, dpi)


def main() -> None:
    args = build_parser().parse_args()

    if args.csv:
        # Single file mode
        generate_radar_charts(
            csv_path=Path(args.csv),
            output_dir=Path(args.output_dir),
            datasets=args.datasets,
            houston_topk=args.houston_topk,
            dpi=args.dpi,
        )
    elif args.without_csv and args.with_csv:
        # Comparison mode
        generate_comparison_radar_charts(
            without_csv=Path(args.without_csv),
            with_csv=Path(args.with_csv),
            output_dir=Path(args.output_dir),
            datasets=args.datasets,
            houston_topk=args.houston_topk,
            dpi=args.dpi,
        )
    else:
        print("Error: Please provide either --csv (single mode) or both --without_csv and --with_csv (comparison mode).")


if __name__ == "__main__":
    main()
