# -*- coding: utf-8 -*-
"""
把 ERS 和 SAM 的分割结果可视化成 PNG 图片。

默认会按当前项目的命名约定读取文件：
- SAM: `SP_mat/<dataset>_<modality>_list.pkl`
- ERS: `SP_mat/<dataset>_<modality>_SPGT_<N>.mat`

其中主模态还兼容旧命名：
- SAM: `SP_mat/<dataset>_list.pkl`
- ERS: `SP_mat/<dataset>_SPGT_<N>.mat`

对双模态数据集，默认会把 `primary` 和 `secondary` 两套结果都画出来。
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

try:
    from segment.segmentation_dataset_utils import (
        DATASET_ALIASES,
        available_modalities,
        dataset_rgb_image,
        resolve_dataset_name,
    )
except ModuleNotFoundError:
    segment_dir = Path(__file__).resolve().parents[1]
    sys.path.append(str(segment_dir))
    from segmentation_dataset_utils import (
        DATASET_ALIASES,
        available_modalities,
        dataset_rgb_image,
        resolve_dataset_name,
    )


def build_parser():
    """定义命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MUUFL', help='Houston, HS-SAR-Berlin, MUUFL')
    parser.add_argument('--data_root', type=str, default='D:/Master_medium/dataset')
    parser.add_argument('--sp_root', type=str, default='./SP_mat')
    parser.add_argument('--output_root', type=str, default='./segment/visualize/outputs')
    parser.add_argument('--modality', type=str, default='both', choices=['primary', 'secondary', 'both'])
    parser.add_argument('--n_segments', type=int, default=None,
                        help='可选：指定要读取哪一个 ERS 分割文件。如果不填，则自动匹配第一个文件')
    parser.add_argument('--seed', type=int, default=7, help='控制伪彩色分割图的随机颜色')
    parser.add_argument('--dpi', type=int, default=180)
    return parser


def _candidate_dataset_prefixes(dataset_name):
    """返回可能对应同一数据集的文件名前缀。"""
    dataset_name = resolve_dataset_name(dataset_name)
    prefixes = [dataset_name]
    for alias, target in DATASET_ALIASES.items():
        if target == dataset_name:
            prefixes.append(alias)
    return list(dict.fromkeys(prefixes))


def _resolve_sam_path(sp_root, dataset_name, modality):
    """根据项目命名规则找到某个模态对应的 SAM 文件。"""
    sp_root = Path(sp_root)
    for prefix in _candidate_dataset_prefixes(dataset_name):
        candidates = [sp_root / f'{prefix}_{modality}_list.pkl']
        if modality == 'primary':
            candidates.append(sp_root / f'{prefix}_list.pkl')
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f'Cannot find SAM masks for dataset {dataset_name} modality {modality} under {sp_root}.'
    )


def _resolve_ers_path(sp_root, dataset_name, modality, n_segments=None):
    """根据项目命名规则找到某个模态对应的 ERS 文件。"""
    sp_root = Path(sp_root)
    for prefix in _candidate_dataset_prefixes(dataset_name):
        if n_segments is not None:
            candidates = [sp_root / f'{prefix}_{modality}_SPGT_{n_segments}.mat']
            if modality == 'primary':
                candidates.append(sp_root / f'{prefix}_SPGT_{n_segments}.mat')
        else:
            candidates = []
            patterns = [f'{prefix}_{modality}_SPGT_*.mat']
            if modality == 'primary':
                patterns.append(f'{prefix}_SPGT_*.mat')
            for pattern in patterns:
                candidates.extend(sorted(sp_root.glob(pattern)))
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f'Cannot find ERS labels for dataset {dataset_name} modality {modality} under {sp_root}.'
    )


def _load_ers_labels(file_path):
    """从 `.mat` 文件中读取超像素标签图。"""
    mat = sio.loadmat(file_path)
    if 'labels' in mat:
        return np.asarray(mat['labels'])
    for key, value in mat.items():
        if not key.startswith('__') and isinstance(value, np.ndarray):
            return np.asarray(value)
    raise KeyError(f'No valid label array was found in {file_path}')


def _load_sam_list(file_path):
    """读取 SAM 自动分割得到的 mask 列表。"""
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def _sam_list_to_label_map(sam_list, shape):
    """
    把 SAM 的 mask 列表转换成二维标签图，便于统一可视化。

    这里会让面积更小的 mask 覆盖更大的 mask，以保留更多细节边界。
    """
    label_map = np.full(shape, -1, dtype=np.int32)
    sorted_masks = sorted(
        sam_list,
        key=lambda item: item.get('area', int(np.sum(item['segmentation']))),
        reverse=True,
    )
    for index, item in enumerate(sorted_masks):
        mask = np.asarray(item['segmentation'], dtype=bool)
        if mask.shape != shape:
            raise ValueError(f'SAM mask shape {mask.shape} does not match image shape {shape}')
        label_map[mask] = index
    return label_map


def _label_boundaries(label_map):
    """从标签图里提取边界像素。"""
    label_map = np.asarray(label_map)
    boundary = np.zeros(label_map.shape, dtype=bool)
    boundary[1:, :] |= label_map[1:, :] != label_map[:-1, :]
    boundary[:-1, :] |= label_map[1:, :] != label_map[:-1, :]
    boundary[:, 1:] |= label_map[:, 1:] != label_map[:, :-1]
    boundary[:, :-1] |= label_map[:, 1:] != label_map[:, :-1]
    return boundary


def _labels_to_color_image(label_map, seed=7):
    """把分割标签图映射成稳定的伪彩色图。"""
    label_map = np.asarray(label_map)
    color_image = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    unique_labels = [label for label in np.unique(label_map) if label >= 0]
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(len(unique_labels), 3), dtype=np.uint8)
    for index, label in enumerate(unique_labels):
        color_image[label_map == label] = palette[index]
    return color_image


def _overlay_boundaries(rgb_image, boundaries, color):
    """在 RGB 图上绘制边界。"""
    overlay = np.array(rgb_image, copy=True)
    overlay[boundaries] = np.array(color, dtype=np.uint8)
    return overlay


def _save_image(image, save_path):
    """保存一张图片。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(save_path, image)


def _save_summary(rgb_image, ers_color, ers_overlay, sam_color, sam_overlay, both_overlay, save_path, dpi):
    """保存一张汇总图，方便整体查看。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=dpi)
    panels = [
        ('RGB', rgb_image),
        ('ERS Color', ers_color),
        ('ERS Overlay', ers_overlay),
        ('SAM Color', sam_color),
        ('SAM Overlay', sam_overlay),
        ('ERS + SAM', both_overlay),
    ]
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def visualize_one_modality(dataset_name, modality, args):
    """可视化某一个模态的 ERS 和 SAM 分割结果。"""
    rgb_image = dataset_rgb_image(dataset_name, modality, args.data_root)
    ers_path = _resolve_ers_path(args.sp_root, dataset_name, modality, args.n_segments)
    sam_path = _resolve_sam_path(args.sp_root, dataset_name, modality)

    ers_labels = _load_ers_labels(ers_path)
    sam_list = _load_sam_list(sam_path)
    sam_labels = _sam_list_to_label_map(sam_list, ers_labels.shape)

    ers_boundary = _label_boundaries(ers_labels)
    sam_boundary = _label_boundaries(sam_labels)

    ers_color = _labels_to_color_image(ers_labels, seed=args.seed)
    sam_color = _labels_to_color_image(sam_labels, seed=args.seed + 97)
    ers_overlay = _overlay_boundaries(rgb_image, ers_boundary, color=(255, 64, 64))
    sam_overlay = _overlay_boundaries(rgb_image, sam_boundary, color=(64, 255, 64))

    both_overlay = np.array(rgb_image, copy=True)
    both_overlay[ers_boundary] = np.array([255, 64, 64], dtype=np.uint8)
    both_overlay[sam_boundary] = np.array([64, 255, 64], dtype=np.uint8)

    output_dir = Path(args.output_root) / resolve_dataset_name(dataset_name) / modality
    _save_image(rgb_image, output_dir / f'{resolve_dataset_name(dataset_name)}_{modality}_rgb.png')
    _save_image(ers_color, output_dir / f'{resolve_dataset_name(dataset_name)}_{modality}_ers_color.png')
    _save_image(ers_overlay, output_dir / f'{resolve_dataset_name(dataset_name)}_{modality}_ers_overlay.png')
    _save_image(sam_color, output_dir / f'{resolve_dataset_name(dataset_name)}_{modality}_sam_color.png')
    _save_image(sam_overlay, output_dir / f'{resolve_dataset_name(dataset_name)}_{modality}_sam_overlay.png')
    _save_image(both_overlay, output_dir / f'{resolve_dataset_name(dataset_name)}_{modality}_combined_overlay.png')
    _save_summary(
        rgb_image,
        ers_color,
        ers_overlay,
        sam_color,
        sam_overlay,
        both_overlay,
        output_dir / f'{resolve_dataset_name(dataset_name)}_{modality}_summary.png',
        dpi=args.dpi,
    )

    print(f'modality={modality}')
    print(f'ers_file={ers_path}')
    print(f'sam_file={sam_path}')
    print(f'output_dir={output_dir}')


def main():
    """脚本入口。"""
    parser = build_parser()
    args = parser.parse_args()

    dataset_name = resolve_dataset_name(args.dataset)
    modalities = available_modalities(dataset_name, args.data_root)
    if args.modality != 'both':
        if args.modality not in modalities:
            raise ValueError(f'Dataset {dataset_name} does not provide modality {args.modality}')
        modalities = [args.modality]

    print(f'dataset={dataset_name}')
    for modality in modalities:
        visualize_one_modality(dataset_name, modality, args)


if __name__ == '__main__':
    main()
