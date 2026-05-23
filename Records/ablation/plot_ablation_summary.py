# -*- coding: utf-8 -*-
"""
根据 D:\Master_medium\Records\ablation 下的结果，绘制：
1. 不同 Margin 下三个数据集的 OA / AA / Kappa 可视化
2. 不同 beta 下三个数据集的 S(OA) / U(OA) / H(OA) / S(AA) / U(AA) / H(AA) 可视化

默认行为：
- 自动读取每个数据集目录下“最新”的实验子目录
- 生成两张总图：
  - margin_summary.png
  - beta_summary.png

直接运行：
    python D:\Master_medium\Records\ablation\plot_ablation_summary.py
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ABLATION_ROOT = Path(r"D:\Master_medium\Records\ablation")
DATASET_ORDER = ["Houston", "HS-SAR-Berlin", "MUUFL"]
DATASET_LABELS = {
    "Houston": "Houston",
    "HS-SAR-Berlin": "Berlin",
    "MUUFL": "MUUFL",
}
DATASET_COLORS = {
    "Houston": "#E07A5F",
    "HS-SAR-Berlin": "#3D5A80",
    "MUUFL": "#81B29A",
}
DATASET_EDGE_COLORS = {
    "Houston": "#B85C38",
    "HS-SAR-Berlin": "#25364D",
    "MUUFL": "#5C8C74",
}


def _get_latest_run_dir(dataset_name: str) -> Path:
    dataset_dir = ABLATION_ROOT / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset ablation directory not found: {dataset_dir}")

    subdirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No ablation run directory found under: {dataset_dir}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _load_csv(dataset_name: str, filename: str) -> pd.DataFrame:
    run_dir = _get_latest_run_dir(dataset_name)
    csv_path = run_dir / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    return pd.read_csv(csv_path)


def _extract_config_value(config_text: str, key: str) -> float:
    config = ast.literal_eval(config_text)
    if key not in config:
        raise KeyError(f"Key '{key}' not found in config: {config_text}")
    return float(config[key])


def _prepare_margin_table() -> pd.DataFrame:
    frames = []
    for dataset in DATASET_ORDER:
        df = _load_csv(dataset, "margin_ablation.csv").copy()
        df["margin"] = df["ablation_config"].map(lambda x: _extract_config_value(x, "margin"))
        df["dataset_name"] = dataset
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["margin", "dataset_name"]).reset_index(drop=True)
    return result


def _prepare_beta_table() -> pd.DataFrame:
    frames = []
    for dataset in DATASET_ORDER:
        df = _load_csv(dataset, "beta_ablation.csv").copy()
        df["beta"] = df["ablation_config"].map(lambda x: _extract_config_value(x, "beta"))
        df["dataset_name"] = dataset
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["beta", "dataset_name"]).reset_index(drop=True)
    return result


def _grouped_bar_positions(num_groups: int, num_series: int, group_width: float = 0.72) -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(num_groups, dtype=float)
    bar_width = group_width / num_series
    return x, x, bar_width


def _plot_grouped_bars(
    ax: plt.Axes,
    x_labels: Sequence[str],
    value_map: Dict[str, Sequence[float]],
    title: str,
    ylabel: str | None = None,
    ylim: Tuple[float, float] | None = None,
    annotate: bool = True,
) -> None:
    datasets = [name for name in DATASET_ORDER if name in value_map]
    x = np.arange(len(x_labels), dtype=float)
    group_width = 0.72
    bar_width = group_width / max(len(datasets), 1)
    start = -group_width / 2 + bar_width / 2

    for idx, dataset in enumerate(datasets):
        values = np.asarray(value_map[dataset], dtype=float)
        pos = x + start + idx * bar_width
        bars = ax.bar(
            pos,
            values,
            width=bar_width,
            color=DATASET_COLORS[dataset],
            edgecolor=DATASET_EDGE_COLORS[dataset],
            linewidth=0.8,
            label=DATASET_LABELS.get(dataset, dataset),
        )
        if annotate:
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ylim[1] - ylim[0]) * 0.01 if ylim else bar.get_height() + 0.3,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=90,
                )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)


def plot_margin_summary(output_path: Path) -> None:
    df = _prepare_margin_table()
    margin_values = sorted(df["margin"].unique().tolist())
    x_labels = [f"{value:g}" for value in margin_values]

    metric_specs = [
        ("OA_mean", "OA (%)", (0, 100)),
        ("AA_mean", "AA (%)", (0, 100)),
        ("Kappa_mean", "Kappa (%)", (0, 100)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    for ax, (column, title, ylim) in zip(axes, metric_specs):
        value_map = {}
        for dataset in DATASET_ORDER:
            subset = df[df["dataset_name"] == dataset].sort_values("margin")
            series = subset[column].astype(float).to_numpy() * 100.0
            value_map[dataset] = series
        _plot_grouped_bars(
            ax=ax,
            x_labels=x_labels,
            value_map=value_map,
            title=title,
            ylabel="Accuracies (×100)" if ax is axes[0] else None,
            ylim=ylim,
        )

    axes[-1].legend(loc="upper right", fontsize=11, frameon=True)
    for ax in axes:
        ax.set_xlabel("Margin", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_beta_summary(output_path: Path) -> None:
    df = _prepare_beta_table()
    beta_values = sorted(df["beta"].unique().tolist())
    x_labels = [f"{value:g}" for value in beta_values]

    metric_specs = [
        ("S_OA_mean", "S(OA) (%)"),
        ("U_OA_mean", "U(OA) (%)"),
        ("H_OA_mean", "H(OA) (%)"),
        ("S_AA_mean", "S(AA) (%)"),
        ("U_AA_mean", "U(AA) (%)"),
        ("H_AA_mean", "H(AA) (%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, (column, title) in zip(axes, metric_specs):
        value_map = {}
        values_all: List[float] = []
        for dataset in DATASET_ORDER:
            subset = df[df["dataset_name"] == dataset].sort_values("beta")
            series = subset[column].astype(float).to_numpy() * 100.0
            value_map[dataset] = series
            values_all.extend(series.tolist())

        ymin = max(0.0, min(values_all) - 2.5)
        ymax = min(100.0, max(values_all) + 2.5)
        if ymax - ymin < 8:
            center = (ymin + ymax) / 2
            ymin = max(0.0, center - 4.5)
            ymax = min(100.0, center + 4.5)

        _plot_grouped_bars(
            ax=ax,
            x_labels=x_labels,
            value_map=value_map,
            title=title,
            ylabel="Accuracies (×100)" if ax in (axes[0], axes[3]) else None,
            ylim=(ymin, ymax),
        )
        ax.set_xlabel(r"$\beta$", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)

    axes[2].legend(loc="upper right", fontsize=11, frameon=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    output_dir = ABLATION_ROOT
    plot_margin_summary(output_dir / "margin_summary.png")
    plot_beta_summary(output_dir / "beta_summary.png")


if __name__ == "__main__":
    main()
