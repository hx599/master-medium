import csv
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from torch.utils.data import Dataset


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
HZSCM_ROOT = PROJECT_ROOT / "HZSCM"
if str(HZSCM_ROOT) not in sys.path:
    sys.path.insert(0, str(HZSCM_ROOT))

from data_read_singlemodal import DATASET_CLASS_NAMES, _load_multimodal_dataset, _resolve_dataset_name  # noqa: E402


DEFAULT_METHOD_PATCH = {
    "mft": 11,
    "ncglf2": 11,
    "exvit": 13,
    "frit": 9,
    "s2crossmamba": 13,
    "udm": 27,
    "unet": 32,
}


@dataclass
class DatasetBundle:
    dataset_name: str
    primary: np.ndarray
    secondary: np.ndarray
    fused: np.ndarray
    gt: np.ndarray
    class_names: list[str]


@dataclass
class SplitBundle:
    train_coords: np.ndarray
    train_labels: np.ndarray
    val_coords: np.ndarray
    val_labels: np.ndarray
    test_coords: np.ndarray
    test_labels: np.ndarray


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_cube(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[:, :, None]
    h, w, c = cube.shape
    reshaped = cube.reshape(h * w, c)
    scaler = preprocessing.StandardScaler()
    reshaped = scaler.fit_transform(reshaped)
    return reshaped.reshape(h, w, c).astype(np.float32)


def project_cube_pca(cube: np.ndarray, out_channels: int) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube for PCA projection, but got shape {cube.shape}.")
    h, w, c = cube.shape
    if c <= out_channels:
        return cube.astype(np.float32)
    reshaped = cube.reshape(h * w, c)
    pca = PCA(n_components=out_channels, svd_solver="full", random_state=0)
    projected = pca.fit_transform(reshaped)
    return projected.reshape(h, w, out_channels).astype(np.float32)


def load_dataset(dataset: str, data_root: str | Path) -> DatasetBundle:
    dataset_name = _resolve_dataset_name(dataset)
    primary, secondary, gt, _ = _load_multimodal_dataset(dataset_name, Path(data_root))
    primary = standardize_cube(primary)
    secondary = standardize_cube(secondary) if secondary is not None else None
    fused = primary if secondary is None else np.concatenate([primary, secondary], axis=2).astype(np.float32)
    gt = np.asarray(gt, dtype=np.int32)
    class_names = DATASET_CLASS_NAMES[dataset_name]
    return DatasetBundle(dataset_name, primary, secondary, fused, gt, class_names)


def _sample_count(num_samples: int, samples_type: str, train_ratio: float, train_num: int | None) -> int:
    if samples_type == "same_num":
        if train_num is None:
            raise ValueError("train_num must be set when samples_type == 'same_num'.")
        return min(int(train_num), num_samples)
    return max(1, int(math.ceil(num_samples * train_ratio)))


def build_known_train_full_test_split(
    gt: np.ndarray,
    del_class: list[int],
    seed: int,
    samples_type: str = "ratio",
    train_ratio: float = 0.05,
    train_num: int | None = None,
    val_ratio: float = 0.1,
) -> SplitBundle:
    rng = random.Random(seed)
    gt = np.asarray(gt, dtype=np.int32)
    nclass = int(gt.max())
    unseen_labels = {cls + 1 for cls in del_class}

    train_coords = []
    train_labels = []
    val_coords = []
    val_labels = []
    test_coords = np.argwhere(gt > 0).astype(np.int32)
    test_labels = gt[test_coords[:, 0], test_coords[:, 1]].astype(np.int64) - 1

    for cls in range(1, nclass + 1):
        if cls in unseen_labels:
            continue
        coords = np.argwhere(gt == cls).astype(np.int32)
        if len(coords) == 0:
            continue
        order = list(range(len(coords)))
        rng.shuffle(order)
        train_count = _sample_count(len(coords), samples_type, train_ratio, train_num)
        remain_after_train = max(0, len(coords) - train_count)
        val_count = min(remain_after_train, int(math.ceil(len(coords) * val_ratio))) if val_ratio > 0 else 0
        train_idx = order[:train_count]
        val_idx = order[train_count:train_count + val_count]
        train_coords.append(coords[train_idx])
        train_labels.append(np.full(len(train_idx), cls - 1, dtype=np.int64))
        if val_count > 0:
            val_coords.append(coords[val_idx])
            val_labels.append(np.full(len(val_idx), cls - 1, dtype=np.int64))

    train_coords = np.concatenate(train_coords, axis=0) if train_coords else np.zeros((0, 2), dtype=np.int32)
    train_labels = np.concatenate(train_labels, axis=0) if train_labels else np.zeros((0,), dtype=np.int64)
    val_coords = np.concatenate(val_coords, axis=0) if val_coords else np.zeros((0, 2), dtype=np.int32)
    val_labels = np.concatenate(val_labels, axis=0) if val_labels else np.zeros((0,), dtype=np.int64)
    return SplitBundle(train_coords, train_labels, val_coords, val_labels, test_coords, test_labels)


def _pad_cube(cube: np.ndarray, patch_size: int) -> np.ndarray:
    pad = patch_size // 2
    return np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def extract_patch_from_padded(padded: np.ndarray, coord: np.ndarray, patch_size: int) -> np.ndarray:
    row, col = int(coord[0]), int(coord[1])
    return padded[row:row + patch_size, col:col + patch_size, :]


class MultiModalPatchDataset(Dataset):
    def __init__(
        self,
        primary: np.ndarray,
        secondary: np.ndarray | None,
        fused: np.ndarray,
        coords: np.ndarray,
        labels: np.ndarray,
        patch_size: int,
    ):
        self.coords = np.asarray(coords, dtype=np.int32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.patch_size = patch_size
        self.primary_padded = _pad_cube(primary, patch_size)
        self.secondary_padded = _pad_cube(secondary, patch_size) if secondary is not None else None
        self.fused_padded = _pad_cube(fused, patch_size)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        coord = self.coords[index]
        primary = extract_patch_from_padded(self.primary_padded, coord, self.patch_size).transpose(2, 0, 1)
        fused = extract_patch_from_padded(self.fused_padded, coord, self.patch_size).transpose(2, 0, 1)
        sample = {
            "primary": torch.from_numpy(primary.astype(np.float32)),
            "fused": torch.from_numpy(fused.astype(np.float32)),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "coord": torch.tensor(coord, dtype=torch.long),
        }
        if self.secondary_padded is not None:
            secondary = extract_patch_from_padded(self.secondary_padded, coord, self.patch_size).transpose(2, 0, 1)
            sample["secondary"] = torch.from_numpy(secondary.astype(np.float32))
        else:
            sample["secondary"] = torch.empty(0, self.patch_size, self.patch_size, dtype=torch.float32)
        return sample


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, nclass: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(nclass)))
    oa = accuracy_score(y_true, y_pred)
    each_acc = np.nan_to_num(np.diag(cm) / np.maximum(cm.sum(axis=1), 1))
    aa = float(each_acc.mean())
    kappa = cohen_kappa_score(y_true, y_pred, labels=list(range(nclass)))
    return {
        "oa": float(oa),
        "aa": aa,
        "kappa": float(kappa),
        "per_class_acc": each_acc,
        "confusion": cm,
    }


def compute_seen_unseen_metrics(per_class_acc: np.ndarray, gt: np.ndarray, del_class: list[int]) -> dict:
    support = np.array([np.sum(gt == i + 1) for i in range(len(per_class_acc))], dtype=np.float64)
    unseen = set(del_class)
    seen_idx = [i for i in range(len(per_class_acc)) if i not in unseen]
    unseen_idx = list(del_class)

    def weighted(indices):
        if not indices:
            return 0.0
        denom = support[indices].sum()
        if denom <= 0:
            return 0.0
        return float(np.sum(per_class_acc[indices] * support[indices]) / denom)

    def avg(indices):
        return float(np.mean(per_class_acc[indices])) if indices else 0.0

    s_oa = weighted(seen_idx)
    u_oa = weighted(unseen_idx)
    s_aa = avg(seen_idx)
    u_aa = avg(unseen_idx)
    h_oa = 0.0 if s_oa + u_oa == 0 else float(2 * s_oa * u_oa / (s_oa + u_oa))
    h_aa = 0.0 if s_aa + u_aa == 0 else float(2 * s_aa * u_aa / (s_aa + u_aa))
    return {
        "s_oa": s_oa,
        "u_oa": u_oa,
        "h_oa": h_oa,
        "s_aa": s_aa,
        "u_aa": u_aa,
        "h_aa": h_aa,
    }


def aggregate_results(results: list[dict]) -> dict:
    keys = ["oa", "aa", "kappa", "s_oa", "u_oa", "h_oa", "s_aa", "u_aa", "h_aa", "train_time", "test_time"]
    agg = {}
    for key in keys:
        values = [r[key] for r in results]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))
    return agg


def save_csv(rows: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def temp_sys_path(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass
