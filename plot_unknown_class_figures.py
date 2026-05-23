#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot figures similar to Fig. 3-13 and Fig. 3-16 in
"未知类高光谱实验结果.pdf".

This script supports two visualization modes:

1. correction
   Compare results without/with label correction.
   It creates one figure per dataset with two subplots:
   left = without correction, right = with correction.

2. beta
   Plot S/U/H curves under different beta values.
   Each input CSV can contain many unseen-class combinations.
   The script averages metrics over all rows of the same dataset.

Examples
--------
1) Plot figures like Fig. 3-13:
python D:\Master_medium\plot_unknown_class_figures.py correction ^
  --without_csv D:\Master_medium\HZSCM\Records\without_correction.csv ^
  --with_csv D:\Master_medium\HZSCM\Records\with_correction.csv ^
  --output_dir D:\Master_medium\visualizations

2) Plot figures like Fig. 3-16:
python D:\Master_medium\plot_unknown_class_figures.py beta ^
  --inputs 0.1=D:\Master_medium\HZSCM\Records\beta_0p1.csv ^
           0.5=D:\Master_medium\HZSCM\Records\beta_0p5.csv ^
           1.0=D:\Master_medium\HZSCM\Records\beta_1p0.csv ^
  --output_dir D:\Master_medium\visualizations
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DATASET_ORDER = ["Houston", "HS-SAR-Berlin", "MUUFL", "Pavia", "Pavia University", "QUH-Qingyun"]
METRIC_CHOICES = ("oa", "aa")
CORRECTION_COLUMNS = {
    "oa": ["S_OA_mean", "U_OA_mean", "H_OA_mean"],
    "aa": ["S_AA_mean", "U_AA_mean", "H_AA_mean"],
}
CORRECTION_LABELS = {
    "oa": ["Seen OA", "Unseen OA", "Harmonic OA"],
    "aa": ["Seen AA", "Unseen AA", "Harmonic AA"],
}
CORRECTION_TITLES = {
    "oa": "Seen / Unseen / Harmonic OA",
    "aa": "Seen / Unseen / Harmonic AA",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot unknown-class hyperspectral result figures similar to Fig. 3-13 and Fig. 3-16."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    correction = subparsers.add_parser(
        "correction",
        help="Compare results without/with label correction.",
    )
    correction.add_argument("--without_csv", type=str, required=True, help="CSV path for results without label correction.")
    correction.add_argument("--with_csv", type=str, required=True, help="CSV path for results with label correction.")
    correction.add_argument("--metric", type=str, default="oa", choices=METRIC_CHOICES, help="Use OA-family or AA-family metrics.")
    correction.add_argument(
        "--sort_by",
        type=str,
        default="unseen_idx",
        choices=["unseen_idx", "without_h", "with_h"],
        help="How to order unseen-class combinations on the x-axis.",
    )
    correction.add_argument("--datasets", nargs="+", default=None, help="Optional dataset filter.")
    correction.add_argument("--dpi", type=int, default=600)
    correction.add_argument("--output_dir", type=str, default=str(Path.cwd() / "visualizations"))

    beta = subparsers.add_parser(
        "beta",
        help="Plot beta parameter curves.",
    )
    beta.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="A list like 0.1=path1.csv 0.5=path2.csv 1.0=path3.csv",
    )
    beta.add_argument("--metric", type=str, default="aa", choices=METRIC_CHOICES, help="Use OA-family or AA-family metrics.")
    beta.add_argument("--datasets", nargs="+", default=None, help="Optional dataset filter.")
    beta.add_argument("--dpi", type=int, default=600)
    beta.add_argument("--output_dir", type=str, default=str(Path.cwd() / "visualizations"))

    return parser


def _read_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "dataset" not in df.columns:
        raise ValueError(f"'dataset' column is missing in {csv_path}")
    if "unseen_idx" in df.columns:
        df["unseen_idx"] = df["unseen_idx"].astype(str)
    if "unseen_names" in df.columns:
        df["unseen_names"] = df["unseen_names"].astype(str)
    return df


def _dataset_sort_key(name: str) -> Tuple[int, str]:
    if name in DATASET_ORDER:
        return DATASET_ORDER.index(name), name
    return len(DATASET_ORDER), name


def _filter_datasets(df: pd.DataFrame, datasets: Optional[Sequence[str]]) -> pd.DataFrame:
    if not datasets:
        return df
    dataset_set = set(datasets)
    return df[df["dataset"].isin(dataset_set)].copy()


def _to_percent(values: Iterable[float]) -> List[float]:
    return [float(v) * 100.0 for v in values]


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str], csv_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_name}: {missing}")


def _safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _compute_ylim(series_list: Sequence[Sequence[float]]) -> Tuple[float, float]:
    merged = [value for series in series_list for value in series if not math.isnan(value)]
    if not merged:
        return 0.0, 100.0
    lower = max(0.0, min(merged) - 5.0)
    upper = min(100.0, max(merged) + 5.0)
    if upper - lower < 15.0:
        center = (upper + lower) / 2.0
        lower = max(0.0, center - 7.5)
        upper = min(100.0, center + 7.5)
    return lower, upper


def plot_label_correction(args: argparse.Namespace) -> None:
    without_df = _filter_datasets(_read_csv(args.without_csv), args.datasets)
    with_df = _filter_datasets(_read_csv(args.with_csv), args.datasets)

    metric_cols = CORRECTION_COLUMNS[args.metric]
    _ensure_columns(without_df, ["dataset", "unseen_idx", *metric_cols], args.without_csv)
    _ensure_columns(with_df, ["dataset", "unseen_idx", *metric_cols], args.with_csv)

    datasets = sorted(set(without_df["dataset"]) & set(with_df["dataset"]), key=_dataset_sort_key)
    if not datasets:
        raise ValueError("No overlapping datasets found between the two CSV files.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harmonic_col = metric_cols[2]
    line_colors = ["#2a9d8f", "#e76f51", "#264653"]
    line_markers = ["o", "s", "^"]

    for dataset in datasets:
        left = without_df[without_df["dataset"] == dataset].copy()
        right = with_df[with_df["dataset"] == dataset].copy()

        merged = pd.merge(
            left[["unseen_idx", "unseen_names", *metric_cols]] if "unseen_names" in left.columns else left[["unseen_idx", *metric_cols]],
            right[["unseen_idx", "unseen_names", *metric_cols]] if "unseen_names" in right.columns else right[["unseen_idx", *metric_cols]],
            on="unseen_idx",
            suffixes=("_without", "_with"),
            how="inner",
        )
        if merged.empty:
            continue

        if args.sort_by == "without_h":
            merged = merged.sort_values(f"{harmonic_col}_without")
        elif args.sort_by == "with_h":
            merged = merged.sort_values(f"{harmonic_col}_with")
        else:
            merged = merged.sort_values("unseen_idx")

        x = list(range(1, len(merged) + 1))
        y_without = [_to_percent(merged[f"{col}_without"]) for col in metric_cols]
        y_with = [_to_percent(merged[f"{col}_with"]) for col in metric_cols]
        ylim = _compute_ylim(y_without + y_with)

        fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True, sharey=True)
        figure_title = f"{dataset} - {CORRECTION_TITLES[args.metric]}"
        fig.suptitle(figure_title, fontsize=18, fontweight="bold")

        for ax, title, values_list in zip(
            axes,
            ["Without Label Correction", "With Label Correction"],
            [y_without, y_with],
        ):
            for values, label, color, marker in zip(values_list, CORRECTION_LABELS[args.metric], line_colors, line_markers):
                ax.plot(
                    x,
                    values,
                    label=label,
                    color=color,
                    marker=marker,
                    linewidth=2.5,
                    markersize=6,
                )
            ax.set_title(title, fontsize=15, fontweight="bold")
            ax.set_xlabel("Unknown-class combination index", fontsize=13)
            ax.set_ylabel("Accuracy (%)", fontsize=13)
            ax.tick_params(axis="both", labelsize=11)
            ax.set_xticks(x)
            ax.set_ylim(*ylim)
            ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
            ax.legend(loc="best", fontsize=11)

        save_path = output_dir / f"{_safe_filename(dataset)}_label_correction_{args.metric}.png"
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


def _parse_beta_inputs(items: Sequence[str]) -> List[Tuple[float, Path]]:
    parsed: List[Tuple[float, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid beta input: {item}. Expected format beta=path.csv")
        beta_str, path_str = item.split("=", 1)
        beta_value = float(beta_str)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        parsed.append((beta_value, path))
    return sorted(parsed, key=lambda x: x[0])


def plot_beta_curves(args: argparse.Namespace) -> None:
    metric_cols = CORRECTION_COLUMNS[args.metric]
    parsed_inputs = _parse_beta_inputs(args.inputs)

    frames: List[pd.DataFrame] = []
    for beta_value, csv_path in parsed_inputs:
        df = _filter_datasets(_read_csv(str(csv_path)), args.datasets)
        _ensure_columns(df, ["dataset", *metric_cols], str(csv_path))
        agg = df.groupby("dataset", as_index=False)[list(metric_cols)].mean()
        agg["beta"] = beta_value
        frames.append(agg)

    if not frames:
        raise ValueError("No valid beta inputs were provided.")

    plot_df = pd.concat(frames, ignore_index=True)
    datasets = sorted(plot_df["dataset"].unique(), key=_dataset_sort_key)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    line_colors = ["#2a9d8f", "#e76f51", "#264653"]
    line_markers = ["o", "s", "^"]

    for dataset in datasets:
        subset = plot_df[plot_df["dataset"] == dataset].sort_values("beta")
        if subset.empty:
            continue

        x = subset["beta"].tolist()
        y_series = [_to_percent(subset[col]) for col in metric_cols]
        ylim = _compute_ylim(y_series)

        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        for values, label, color, marker in zip(y_series, CORRECTION_LABELS[args.metric], line_colors, line_markers):
            ax.plot(
                x,
                values,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.5,
                markersize=8,
            )

        ax.set_title(f"{dataset} - beta analysis", fontsize=16, fontweight="bold")
        ax.set_xlabel("beta", fontsize=14)
        ax.set_ylabel("Accuracy (%)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xticks(x)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(loc="best", fontsize=12)

        save_path = output_dir / f"{_safe_filename(dataset)}_beta_{args.metric}.png"
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "correction":
        plot_label_correction(args)
    elif args.mode == "beta":
        plot_beta_curves(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
