# -*- coding: utf-8 -*-
"""
对比方法（ExViT / MFT）的 per-class 结果记录脚本。
参考 HZSCM/run_per_class_results_presets.py 的输出格式。

预设组合（0-based 类别编号）:
    Houston:     1/9, 7/3, 10/14
    MUUFL:       0/5, 1/9, 3/10
    HS-SAR-Berlin: 4/5, 4/7, 0/5

输出内容:
    1. 每个组合一份 per-class CSV
    2. 一份总汇总 CSV
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hzscm_common import (
    DEFAULT_METHOD_PATCH,
    THIS_DIR,
    MultiModalPatchDataset,
    aggregate_results,
    build_known_train_full_test_split,
    compute_metrics,
    compute_seen_unseen_metrics,
    load_dataset,
    project_cube_pca,
    set_seed,
    temp_sys_path,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
METHODS: List[str] = ["exvit", "mft"]
NUM_REPEATS = 5
BATCH_SIZE = 256
LR = 5e-4
WEIGHT_DECAY = 0.0
SEED = 0
DATA_ROOT = r"D:\Master_medium\dataset"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
OUTPUT_DIR = Path(r"D:\Master_medium\CompareMethods\Records\per_class_presets")

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

# ---------------------------------------------------------------------------
# 从 run_hzscm_baseline.py 复用的工具函数
# ---------------------------------------------------------------------------

import importlib.util


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _default_epochs(method: str) -> int:
    return {"mft": 120, "exvit": 200}[method]


def _build_method(method: str, bundle, patch_size: int, nclass: int):
    """构建模型、优化器、调度器以及训练/推理函数（与 run_hzscm_baseline.build_method 一致）。"""
    primary_channels = bundle.primary.shape[2]
    secondary_channels = bundle.secondary.shape[2]

    if method == "mft":
        from mft_model_hzscm import MFT

        model = MFT(
            fm=16,
            num_hsi_bands=primary_channels,
            num_aux_bands=secondary_channels,
            num_classes=nclass,
            patch_size=patch_size,
        )
        criterion = nn.CrossEntropyLoss()

        def train_loss_fn(batch, model, device):
            x1 = batch["primary"].to(device).flatten(2)
            x2 = batch["secondary"].to(device).flatten(2)
            labels = batch["label"].to(device)
            logits = model(x1, x2)
            return criterion(logits, labels), logits

        def eval_logits_fn(batch, model, device):
            return model(
                batch["primary"].to(device).flatten(2),
                batch["secondary"].to(device).flatten(2),
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        return model, optimizer, scheduler, train_loss_fn, eval_logits_fn

    if method == "exvit":
        exvit_module = _load_module(
            "exvit_model_hzscm", THIS_DIR / "ExViT" / "MViT_pytorch_upload.py"
        )
        model = exvit_module.MViT(
            patch_size=patch_size,
            num_patches=[primary_channels, secondary_channels],
            num_classes=nclass,
            dim=64,
            depth=6,
            heads=4,
            mlp_dim=32,
            dropout=0.1,
            emb_dropout=0.1,
            mode="MViT",
        )
        criterion = nn.CrossEntropyLoss()

        def train_loss_fn(batch, model, device):
            labels = batch["label"].to(device)
            logits = model(batch["primary"].to(device), batch["secondary"].to(device))
            return criterion(logits, labels), logits

        def eval_logits_fn(batch, model, device):
            return model(batch["primary"].to(device), batch["secondary"].to(device))

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        return model, optimizer, scheduler, train_loss_fn, eval_logits_fn

    raise ValueError(f"Unknown method: {method}")


def _run_epoch(model, loader, optimizer, train_loss_fn, device):
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        optimizer.zero_grad()
        loss, logits = train_loss_fn(batch, model, device)
        loss.backward()
        optimizer.step()
        batch_size = logits.shape[0]
        total_loss += loss.item() * batch_size
        total += batch_size
    return total_loss / max(total, 1)


@torch.no_grad()
def _predict_logits(model, loader, eval_logits_fn, device):
    model.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        logits = eval_logits_fn(batch, model, device)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])
    return torch.cat(all_logits, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


# ---------------------------------------------------------------------------
# 单次实验（返回 per-class 精度）
# ---------------------------------------------------------------------------


def _run_single_repeat_per_class(
    method: str,
    dataset: str,
    del_class: List[int],
    repeat_idx: int,
    device: torch.device,
) -> Dict:
    """运行一次实验，返回包含 per_class_acc (pre/final) 的字典。"""
    seed = SEED + repeat_idx
    set_seed(seed)

    bundle = load_dataset(dataset, DATA_ROOT)
    split = build_known_train_full_test_split(
        bundle.gt,
        del_class=del_class,
        seed=seed,
        train_ratio=0.05,
        val_ratio=0.1,
    )

    patch_size = DEFAULT_METHOD_PATCH[method]
    nclass = int(bundle.gt.max())

    primary_cube = bundle.primary
    if method == "s2crossmamba":
        primary_cube = project_cube_pca(primary_cube, out_channels=30)
    fused_cube = np.concatenate([primary_cube, bundle.secondary], axis=2).astype(np.float32)

    train_set = MultiModalPatchDataset(
        primary_cube, bundle.secondary, fused_cube,
        split.train_coords, split.train_labels, patch_size,
    )
    test_set = MultiModalPatchDataset(
        primary_cube, bundle.secondary, fused_cube,
        split.test_coords, split.test_labels, patch_size,
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    method_bundle = SimpleNamespace(
        dataset_name=bundle.dataset_name,
        primary=primary_cube,
        secondary=bundle.secondary,
        fused=fused_cube,
        gt=bundle.gt,
        class_names=bundle.class_names,
    )
    model, optimizer, scheduler, train_loss_fn, eval_logits_fn = _build_method(
        method, method_bundle, patch_size, nclass
    )
    model = model.to(device)

    # ---- 训练前评估 ----
    pre_logits, pre_labels = _predict_logits(model, test_loader, eval_logits_fn, device)
    pre_pred = pre_logits.argmax(axis=1)
    pre_metrics = compute_metrics(pre_labels, pre_pred, nclass)

    # ---- 训练 ----
    epochs = _default_epochs(method)
    tic_train = time.time()
    for epoch_idx in range(epochs):
        _run_epoch(model, train_loader, optimizer, train_loss_fn, device)
        if scheduler is not None:
            scheduler.step()
    train_time = time.time() - tic_train

    # ---- 训练后评估 ----
    tic_test = time.time()
    post_logits, post_labels = _predict_logits(model, test_loader, eval_logits_fn, device)
    post_pred = post_logits.argmax(axis=1)
    test_time = time.time() - tic_test

    post_metrics = compute_metrics(post_labels, post_pred, nclass)
    su_metrics = compute_seen_unseen_metrics(post_metrics["per_class_acc"], bundle.gt, del_class)
    pre_su_metrics = compute_seen_unseen_metrics(pre_metrics["per_class_acc"], bundle.gt, del_class)

    return {
        "nclass": nclass,
        # 训练前总指标
        "pre_oa": pre_metrics["oa"],
        "pre_aa": pre_metrics["aa"],
        "pre_kappa": pre_metrics["kappa"],
        "pre_s_oa": pre_su_metrics["s_oa"],
        "pre_u_oa": pre_su_metrics["u_oa"],
        "pre_h_oa": pre_su_metrics["h_oa"],
        "pre_s_aa": pre_su_metrics["s_aa"],
        "pre_u_aa": pre_su_metrics["u_aa"],
        "pre_h_aa": pre_su_metrics["h_aa"],
        # 训练后总指标
        "oa": post_metrics["oa"],
        "aa": post_metrics["aa"],
        "kappa": post_metrics["kappa"],
        "s_oa": su_metrics["s_oa"],
        "u_oa": su_metrics["u_oa"],
        "h_oa": su_metrics["h_oa"],
        "s_aa": su_metrics["s_aa"],
        "u_aa": su_metrics["u_aa"],
        "h_aa": su_metrics["h_aa"],
        "train_time": train_time,
        "test_time": test_time,
        # per-class
        "pre_ac": pre_metrics["per_class_acc"],
        "ac": post_metrics["per_class_acc"],
    }


# ---------------------------------------------------------------------------
# 聚合 & 输出（与 HZSCM/run_per_class_results.py 对齐）
# ---------------------------------------------------------------------------

# 从 HZSCM 子系统导入类名映射
sys.path.insert(0, str(THIS_DIR.parent / "HZSCM"))
from data_read_singlemodal import DATASET_CLASS_NAMES  # noqa: E402


def _aggregate_scalar(results: List[Dict]) -> Dict[str, float]:
    keys_scalar = [
        "pre_oa", "pre_aa", "pre_kappa",
        "pre_s_oa", "pre_u_oa", "pre_h_oa",
        "pre_s_aa", "pre_u_aa", "pre_h_aa",
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


def _build_summary_row(
    method: str,
    dataset: str,
    del_class: List[int],
    results: List[Dict],
) -> Dict[str, str]:
    nclass = int(results[0]["nclass"])
    class_names = DATASET_CLASS_NAMES.get(dataset, [f"class_{i}" for i in range(nclass)])
    unseen_names = [class_names[i] for i in del_class]

    row: Dict[str, str] = {
        "method": method,
        "dataset": dataset,
        "unseen_idx": ";".join(str(x) for x in del_class),
        "unseen_names": ";".join(unseen_names),
        "nclass": str(nclass),
        "num_repeats": str(NUM_REPEATS),
    }

    scalar_agg = _aggregate_scalar(results)
    mapped_names = {
        "pre_oa": "pre_OA",
        "pre_aa": "pre_AA",
        "pre_kappa": "pre_Kappa",
        "pre_s_oa": "pre_S_OA",
        "pre_u_oa": "pre_U_OA",
        "pre_h_oa": "pre_H_OA",
        "pre_s_aa": "pre_S_AA",
        "pre_u_aa": "pre_U_AA",
        "pre_h_aa": "pre_H_AA",
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


def _print_per_class_summary(row: Dict[str, str], dataset: str, method: str) -> None:
    nclass = int(row["nclass"])
    print(f"\n[{method}] {dataset}  unseen={row['unseen_idx']}")
    print("Per-class final test accuracy:")
    print("-" * 72)
    print(f"{'class':>5s}  {'type':>7s}  {'name':<30s}  {'final_acc':>16s}")
    print("-" * 72)
    for idx in range(1, nclass + 1):
        prefix = f"class_{idx}"
        kind = "unseen" if row[f"{prefix}_is_unseen"] == "1" else "seen"
        name = row[f"{prefix}_name"]
        acc = (
            f"{float(row[f'{prefix}_final_acc_mean']):.4f}"
            f" +/- {float(row[f'{prefix}_final_acc_std']):.4f}"
        )
        print(f"{idx:5d}  {kind:>7s}  {name:<30.30s}  {acc:>16s}")
    print("-" * 72)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _run_one_experiment(
    method: str,
    dataset: str,
    del_class: List[int],
    device: torch.device,
) -> Dict[str, str]:
    print(f"\n{'=' * 80}")
    print(f"Method: {method}  |  Dataset: {dataset}")
    print(f"Unseen classes (0-based): {del_class}  |  Repeats: {NUM_REPEATS}")
    print(f"{'=' * 80}")

    results: List[Dict] = []
    for repeat_idx in range(NUM_REPEATS):
        print(f"\n--- Repeat {repeat_idx + 1}/{NUM_REPEATS} ---")
        metrics = _run_single_repeat_per_class(method, dataset, del_class, repeat_idx, device)
        results.append(metrics)

    output_row = _build_summary_row(method, dataset, del_class, results)

    pair_str = "_".join(str(x) for x in del_class)
    output_path = OUTPUT_DIR / f"{method}_{dataset}_per_class_{pair_str}.csv"
    _write_single_row_csv(output_row, output_path)
    _print_per_class_summary(output_row, dataset, method)
    print(f"Saved per-class result CSV to: {output_path}")
    return output_row


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)

    all_rows: List[Dict[str, str]] = []
    for method in METHODS:
        for dataset, del_class in EXPERIMENT_PRESETS:
            row = _run_one_experiment(method, dataset, del_class, device)
            all_rows.append(row)

    summary_path = OUTPUT_DIR / "all_methods_per_class_summary.csv"
    _write_summary_csv(all_rows, summary_path)
    print(f"\nSaved summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
