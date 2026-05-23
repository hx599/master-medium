# -*- coding: utf-8 -*-
"""
分割辅助脚本共用的工具函数。

主要负责三件事：
1. 按数据集名字读取主模态、辅助模态和 GT；
2. 把不同模态转成适合 SAM / ERS 使用的 RGB 图；
3. 统一生成输出路径，并把超像素标签保存成项目约定的 `.mat` 格式。
"""

from pathlib import Path

import numpy as np
import scipy.io as sio


DATASET_ALIASES = {
    'Berlin': 'HS-SAR-Berlin',
    'MuUFL': 'MUUFL',
}

DATASET_MODALITY_BANDS = {
    'Houston': {'primary': [59, 40, 23], 'secondary': None},
    'HS-SAR-Berlin': {'primary': [37, 19, 3], 'secondary': None},
    'MUUFL': {'primary': [25, 14, 6], 'secondary': None},
}


def resolve_dataset_name(dataset_name):
    """把别名统一映射成项目内部使用的数据集名。"""
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def load_mat_array(file_path, preferred_keys=None):
    """从 `.mat` 文件中读取数组，优先使用指定键名。"""
    mat = sio.loadmat(file_path)
    if preferred_keys is not None:
        for key in preferred_keys:
            if key in mat:
                return mat[key]
    for key, value in mat.items():
        if not key.startswith('__') and isinstance(value, np.ndarray):
            return value
    raise KeyError(f'no valid array was found in {file_path}')


def load_dataset_modalities(dataset_name, data_root):
    """按数据集名称读取主模态、第二模态和标签。"""
    dataset_name = resolve_dataset_name(dataset_name)
    data_root = Path(data_root)

    if dataset_name == 'Houston':
        primary = load_mat_array(data_root / 'Houston' / 'data_HS_LR.mat', ['data_HS_LR'])
        secondary = load_mat_array(data_root / 'Houston' / 'data_MS_HR.mat', ['data_MS_HR', 'LiDAR'])
        gt = load_mat_array(data_root / 'Houston' / 'Houston_label.mat', ['label'])
        return primary, secondary, gt

    if dataset_name == 'HS-SAR-Berlin':
        primary = load_mat_array(data_root / 'HS-SAR-Berlin' / 'data_HS_LR.mat', ['data_HS_LR'])
        secondary = load_mat_array(data_root / 'HS-SAR-Berlin' / 'data_SAR_HR.mat', ['data_SAR_HR'])
        gt = load_mat_array(data_root / 'HS-SAR-Berlin' / 'Berlin_groundtruth.mat', ['label'])
        return primary, secondary, gt

    if dataset_name == 'MUUFL':
        primary = load_mat_array(data_root / 'MUUFL' / 'MuUFL_hsi.mat', ['hsi'])
        secondary = load_mat_array(data_root / 'MUUFL' / 'MUUFL_LiDAR.mat', ['lidar_data'])
        gt = load_mat_array(data_root / 'MUUFL' / 'MuUFL_label.mat', ['label'])
        return primary, secondary, gt

    raise ValueError(f'Unsupported dataset for segmentation utility: {dataset_name}')


def available_modalities(dataset_name, data_root):
    """
    返回当前数据集实际可用的模态列表。

    对双模态数据集会返回 `['primary', 'secondary']`，
    对单模态数据集只返回 `['primary']`。
    """
    dataset_name = resolve_dataset_name(dataset_name)
    _, secondary, _ = load_dataset_modalities(dataset_name, data_root)
    modalities = ['primary']
    if secondary is not None:
        modalities.append('secondary')
    return modalities


def ensure_3d_cube(image):
    """保证输入是 `(H, W, C)` 形式，单通道则补成三维。"""
    image = np.asarray(image)
    if image.ndim == 2:
        return image[:, :, None]
    if image.ndim == 3:
        return image
    raise ValueError(f'Expected a 2D or 3D array, but got shape {image.shape}')


def to_rgb_image(image, rgb_bands=None):
    """
    把多光谱/高光谱/单通道数据转成 3 通道 RGB 图。

    这里的 RGB 仅用于 SAM / ERS 等图像分割工具，不参与模型训练。
    """
    image = ensure_3d_cube(image).astype(np.float32)
    channel_count = image.shape[2]

    if channel_count == 1:
        # 单通道数据直接复制三份，变成伪 RGB。
        rgb = np.repeat(image, 3, axis=2)
    elif channel_count == 2:
        # 双通道数据补一个均值通道，凑成三通道。
        rgb = np.concatenate([image, np.mean(image, axis=2, keepdims=True)], axis=2)
    else:
        # 多通道数据则抽取三个代表性波段作为可视化输入。
        if rgb_bands is None:
            rgb_bands = np.linspace(0, channel_count - 1, 3).round().astype(int).tolist()
        rgb_bands = [min(max(int(idx), 0), channel_count - 1) for idx in rgb_bands[:3]]
        while len(rgb_bands) < 3:
            rgb_bands.append(rgb_bands[-1])
        rgb = image[:, :, rgb_bands]

    # 做线性归一化，映射到 0~255 的 uint8 图像。
    rgb = rgb - np.min(rgb, axis=(0, 1), keepdims=True)
    rgb_max = np.max(rgb, axis=(0, 1), keepdims=True)
    rgb_max[rgb_max < 1e-6] = 1.0
    return np.clip(rgb / rgb_max * 255.0, 0, 255).astype(np.uint8)


def dataset_rgb_image(dataset_name, modality, data_root):
    """读取指定模态，并转成 RGB 图供外部分割工具使用。"""
    dataset_name = resolve_dataset_name(dataset_name)
    primary, secondary, _ = load_dataset_modalities(dataset_name, data_root)
    band_cfg = DATASET_MODALITY_BANDS.get(dataset_name, {})
    if modality == 'primary':
        return to_rgb_image(primary, band_cfg.get('primary'))
    if modality == 'secondary':
        if secondary is None:
            raise ValueError(f'Dataset {dataset_name} has no secondary modality')
        return to_rgb_image(secondary, band_cfg.get('secondary'))
    raise ValueError(f'Unsupported modality: {modality}')


def default_sam_output_path(dataset_name, sp_root, modality=None):
    """返回项目约定的 SAM 输出路径。"""
    dataset_name = resolve_dataset_name(dataset_name)
    if modality is None:
        return Path(sp_root) / f'{dataset_name}_list.pkl'
    return Path(sp_root) / f'{dataset_name}_{modality}_list.pkl'


def default_ers_output_path(dataset_name, sp_root, n_segments, modality=None):
    """返回项目约定的 ERS 输出路径。"""
    dataset_name = resolve_dataset_name(dataset_name)
    if modality is None:
        return Path(sp_root) / f'{dataset_name}_SPGT_{n_segments}.mat'
    return Path(sp_root) / f'{dataset_name}_{modality}_SPGT_{n_segments}.mat'


def save_superpixel_labels(labels, output_path):
    """把超像素标签保存成 `.mat`，变量名固定为 `labels`。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(output_path, {'labels': labels.astype(np.int32)})
