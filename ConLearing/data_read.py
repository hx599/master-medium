# -*- coding: utf-8 -*-
"""
数据读取模块 - 持续学习版本。

基于 HZSCM 的 data_read_singlemodal.py 扩展：
1. 支持增量学习场景的数据划分
2. 支持加载预训练的教师模型进行蒸馏
3. 支持多模态数据的模态适配器处理
"""

from pathlib import Path
import pickle
import random
import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None

# 从 util 导入需要的函数
import sys
sys.path.append('D:/Master_medium/ConLearing')
from util import (
    SpiltHSI,
    get_pseudo_labels,
    get_pseudo_labels_using_SAM_and_ERS,
    get_multimodal_pseudo_labels_with_individual_segments,
    get_multimodal_pseudo_labels_using_individual_SAM_and_ERS,
)

clip_model_name = ['_rsicd', '_rgb', '_rsicd_rgb']

DATASET_ALIASES = {
    'Berlin': 'HS-SAR-Berlin',
    'MuUFL': 'MUUFL',
}

DATASET_MODALITY_BANDS = {
    'Houston': {'primary': [59, 40, 23], 'secondary': None},
    'HS-SAR-Berlin': {'primary': [37, 19, 3], 'secondary': [0, 1, 2]},
    'MUUFL': {'primary': [25, 14, 6], 'secondary': None},
    'Pavia': {'primary': [55, 31, 12]},
    'PaviaR': {'primary': [55, 31, 12]},
    'QUH_Pingan': {'primary': [40, 20, 10]},
    'QUH_Qingyun': {'primary': [40, 20, 10]},
    'CASI': {'primary': [59, 40, 23]},
    'CASIR': {'primary': [59, 40, 23]},
    'CASI2018': {'primary': [59, 40, 23]},
}

DATASET_CLASS_NAMES = {
    'Houston': [
        "healthy grass", "stressed grass", "synthetic grass", "trees", "soil",
        "water", "residential buildings", "commercial buildings", "road", "highway",
        "railway", "the first kind of parking lot", "the second kind of parking lot",
        "tennis court", "running track"
    ],
    'HS-SAR-Berlin': [
        "forest", "residential area", "industrial area", "low plants",
        "soil", "allotment", "commercial area", "water"
    ],
    'MUUFL': [
        "trees", "mostly grass", "mixed ground surface", "dirt or sand", "road",
        "water", "building shadow", "buildings", "sidewalk", "yellow curb", "cloth panels"
    ],
    'Pavia': [
        "asphalt", "grass", "gravel", "trees", "metal sheet roof",
        "soil", "bitumen", "bricks", "shadow"
    ],
    'PaviaR': [
        "road", "grass", "gravel", "trees", "metal sheet roof",
        "soil", "bitumen", "shadow"
    ],
}

DATASET_NCLASS = {
    'Houston': 15,
    'HS-SAR-Berlin': 8,
    'MUUFL': 11,
    'Pavia': 9,
    'PaviaR': 8,
    'QUH_Pingan': 10,
    'QUH_Qingyun': 6,
    'CASI': 15,
    'CASIR': 12,
    'CASI2018': 13,
}


def _resolve_dataset_name(dataset_name):
    """把数据集别名转换成项目内部统一使用的名称。"""
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def _candidate_dataset_prefixes(dataset_name):
    """返回可能对应同一数据集的文件名前缀。"""
    prefixes = [dataset_name]
    for alias, target in DATASET_ALIASES.items():
        if target == dataset_name:
            prefixes.append(alias)
    return list(dict.fromkeys(prefixes))


def get_RGB(image, RGB_list=None, aug=False):
    """把任意模态数据转成 3 通道图像。"""
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError(f'expected a 2D or 3D array, but got shape {image.shape}')

    image = image.astype(np.float32)
    channel_count = image.shape[2]

    if channel_count == 1:
        rgb = np.repeat(image, 3, axis=2)
    elif channel_count == 2:
        third_channel = np.mean(image, axis=2, keepdims=True)
        rgb = np.concatenate([image, third_channel], axis=2)
    else:
        if RGB_list is None:
            RGB_list = np.linspace(0, channel_count - 1, 3).round().astype(int).tolist()
        RGB_list = [min(max(int(idx), 0), channel_count - 1) for idx in RGB_list[:3]]
        while len(RGB_list) < 3:
            RGB_list.append(RGB_list[-1])
        rgb = image[:, :, RGB_list]

    rgb = rgb - np.min(rgb, axis=(0, 1), keepdims=True)
    rgb_max = np.max(rgb, axis=(0, 1), keepdims=True)
    rgb_max[rgb_max < 1e-6] = 1.0
    rgb = np.clip(rgb / rgb_max * 255.0, 0, 255).astype(np.uint8)

    if aug and cv2 is not None:
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
        for channel in range(3):
            rgb[:, :, channel] = clahe.apply(rgb[:, :, channel])

    return rgb


def _load_mat_array(file_path, preferred_keys=None):
    """从 .mat 文件读取数组。"""
    mat = sio.loadmat(file_path)
    if preferred_keys is not None:
        for key in preferred_keys:
            if key in mat:
                return mat[key]
    for key, value in mat.items():
        if not key.startswith('__') and isinstance(value, np.ndarray):
            return value
    raise KeyError(f'no valid array was found in {file_path}')


def _load_multimodal_dataset(dataset_name, data_root):
    """读取双模态数据集。"""
    if dataset_name == 'Houston':
        data = _load_mat_array(data_root / 'Houston' / 'data_HS_LR.mat', ['data_HS_LR'])
        aux_data = _load_mat_array(data_root / 'Houston' / 'data_MS_HR.mat', ['data_MS_HR', 'LiDAR'])
        gt = _load_mat_array(data_root / 'Houston' / 'Houston_label.mat', ['label'])
        attri_descriptions = DATASET_CLASS_NAMES['Houston']
        return data, aux_data, gt, attri_descriptions

    if dataset_name == 'HS-SAR-Berlin':
        data = _load_mat_array(data_root / 'HS-SAR-Berlin' / 'data_HS_LR.mat', ['data_HS_LR'])
        aux_data = _load_mat_array(data_root / 'HS-SAR-Berlin' / 'data_SAR_HR.mat', ['data_SAR_HR'])
        gt = _load_mat_array(data_root / 'HS-SAR-Berlin' / 'Berlin_groundtruth.mat', ['label'])
        attri_descriptions = DATASET_CLASS_NAMES['HS-SAR-Berlin']
        return data, aux_data, gt, attri_descriptions

    if dataset_name == 'MUUFL':
        data = _load_mat_array(data_root / 'MUUFL' / 'MuUFL_hsi.mat', ['hsi'])
        aux_data = _load_mat_array(data_root / 'MUUFL' / 'MUUFL_LiDAR.mat', ['lidar_data'])
        gt = _load_mat_array(data_root / 'MUUFL' / 'MuUFL_label.mat', ['label'])
        attri_descriptions = DATASET_CLASS_NAMES['MUUFL']
        return data, aux_data, gt, attri_descriptions

    return None


def _load_sp_gt(sp_root, dataset_name, modality='primary'):
    """读取超像素标签图。"""
    for prefix in _candidate_dataset_prefixes(dataset_name):
        patterns = [f'{prefix}_{modality}_SPGT_*.mat']
        if modality == 'primary':
            patterns.append(f'{prefix}_SPGT_*.mat')
        for pattern in patterns:
            matches = sorted(sp_root.glob(pattern))
            for mat_path in matches:
                mat = sio.loadmat(mat_path)
                if 'labels' in mat:
                    return mat['labels']
    expected = f'{dataset_name}_{modality}_SPGT_*.mat'
    if modality == 'primary':
        expected += f' or legacy {dataset_name}_SPGT_*.mat'
    raise FileNotFoundError(f'Cannot find superpixel file under {sp_root} for {dataset_name}')


def _load_sam_list(sp_root, dataset_name, modality='primary'):
    """读取 SAM mask 列表。"""
    for prefix in _candidate_dataset_prefixes(dataset_name):
        candidates = [sp_root / f'{prefix}_{modality}_list.pkl']
        if modality == 'primary':
            candidates.append(sp_root / f'{prefix}_list.pkl')
        for candidate in candidates:
            if candidate.exists():
                with open(candidate, 'rb') as f:
                    return pickle.load(f)
    expected = f'{dataset_name}_{modality}_list.pkl'
    if modality == 'primary':
        expected += f' or legacy {dataset_name}_list.pkl'
    raise FileNotFoundError(f'Cannot find SAM list under {sp_root} for {dataset_name}')


def _get_clip_dirs(args):
    """根据 multi_clip 配置决定启用哪些 CLIP 权重。"""
    clip_dirs = []
    if args.multi_clip in (0, 2):
        clip_dirs.append(args.rsicd_clip_dir)
    if args.multi_clip in (1, 2):
        clip_dirs.append(args.rgb_clip_dir)
    if len(clip_dirs) == 0:
        raise ValueError(f'Unsupported multi_clip value: {args.multi_clip}')
    return clip_dirs


def _build_modality_images(dataset_name, data, aux_data, use_multimodal):
    """把一个或多个模态都转换成 CLIP 可接受的 RGB 图像。"""
    band_cfg = DATASET_MODALITY_BANDS.get(dataset_name, {})
    images = [get_RGB(data, band_cfg.get('primary'))]
    if use_multimodal and aux_data is not None:
        images.append(get_RGB(aux_data, band_cfg.get('secondary')))
    return images


def _pseudo_label_cache_path(sp_root, dataset_name, args, multimodal):
    """构造伪标签缓存文件名。"""
    suffix = clip_model_name[args.multi_clip]
    if multimodal:
        filename = f'{dataset_name}_margin{args.margin}_usesam{args.use_sam}{suffix}_permodality_avg_pseudo_labels.mat'
    else:
        filename = f'{dataset_name}_margin{args.margin}_usesam{args.use_sam}{suffix}_pseudo_labels.mat'
    return sp_root / filename


def _load_or_generate_pseudo_labels(args, dataset_name, data, aux_data, attri_descriptions, sp_gt):
    """优先从磁盘读取伪标签，或在线生成并缓存。"""
    sp_root = Path(args.sp_root)
    use_multimodal = bool(args.use_multimodal_pseudo) and aux_data is not None

    cache_path = _pseudo_label_cache_path(sp_root, dataset_name, args, use_multimodal)
    single_cache_path = _pseudo_label_cache_path(sp_root, dataset_name, args, False)

    if args.pseudo_label_mode == 'load':
        if cache_path.exists():
            return sio.loadmat(cache_path)['pseudo_labels']
        if not use_multimodal and single_cache_path.exists():
            return sio.loadmat(single_cache_path)['pseudo_labels']
        raise FileNotFoundError(f'Cannot find pseudo labels for {dataset_name}. Set --pseudo_label_mode generate.')

    if len(attri_descriptions) == 0:
        raise ValueError(f'No class descriptions for {dataset_name}')
    if CLIPModel is None or CLIPProcessor is None:
        raise ImportError('transformers is required to generate pseudo labels.')

    modality_images = _build_modality_images(dataset_name, data, aux_data, use_multimodal)
    modality_sp_gts = [sp_gt]
    if use_multimodal:
        modality_sp_gts.append(_load_sp_gt(sp_root, dataset_name, modality='secondary'))

    modality_sam_lists = None
    if args.use_sam:
        modality_sam_lists = [_load_sam_list(sp_root, dataset_name, modality='primary')]
        if use_multimodal:
            modality_sam_lists.append(_load_sam_list(sp_root, dataset_name, modality='secondary'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_outputs = []

    with torch.no_grad():
        for clip_dir in _get_clip_dirs(args):
            clip_model = CLIPModel.from_pretrained(clip_dir).to(device)
            processor = CLIPProcessor.from_pretrained(clip_dir)

            if args.use_sam and use_multimodal:
                pseudo_probs = get_multimodal_pseudo_labels_using_individual_SAM_and_ERS(
                    modality_sp_gts, modality_sam_lists, modality_images, clip_model,
                    processor, attri_descriptions, margin=args.margin)
            elif args.use_sam:
                pseudo_probs = get_pseudo_labels_using_SAM_and_ERS(
                    modality_sp_gts[0], modality_sam_lists[0], modality_images[0],
                    clip_model, processor, attri_descriptions, margin=args.margin)
            elif use_multimodal:
                pseudo_probs = get_multimodal_pseudo_labels_with_individual_segments(
                    modality_sp_gts, modality_images, clip_model, processor,
                    attri_descriptions, margin=args.margin)
            else:
                pseudo_probs = get_pseudo_labels(
                    modality_sp_gts[0], modality_images[0], clip_model, processor,
                    attri_descriptions, margin=args.margin)

            clip_outputs.append(pseudo_probs.cpu())
            del clip_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pseudo_labels = torch.stack(clip_outputs, dim=0).mean(dim=0).numpy().astype(np.float32)
    save_path = cache_path if use_multimodal else single_cache_path
    sio.savemat(save_path, {'pseudo_labels': pseudo_labels})
    return pseudo_labels


def read_data(args, curr_seed, del_class=None, incremental_classes=None):
    """
    读取数据集，返回训练主流程需要的全部内容。

    新增参数:
        del_class: 零样本设定中作为unseen的类别 (0-based)
        incremental_classes: 持续学习中新引入的类别 (0-based)
    """
    dataset_name = _resolve_dataset_name(args.dataset)
    multimodal_dataset = _load_multimodal_dataset(dataset_name, Path(args.data_root))
    aux_data = None

    if multimodal_dataset is not None:
        data, aux_data, gt, attri_descriptions = multimodal_dataset
    else:
        legacy_root = Path(args.data_root)
        if not legacy_root.exists():
            legacy_root = Path('F:/HSIdata')

        if dataset_name == 'Pavia':
            data_file = legacy_root / 'Pavia' / 'Pavia.mat'
            gt_file = legacy_root / 'Pavia' / 'Pavia_groundtruth.mat'
            data_mat = sio.loadmat(data_file)
            gt_mat = sio.loadmat(gt_file)
            data = data_mat['paviaU']
            gt = gt_mat['paviaU_gt']
            attri_descriptions = DATASET_CLASS_NAMES['Pavia']
        else:
            raise Exception(f'dataset {dataset_name} not found in legacy data')

    sp_root = Path(args.sp_root)
    SP_gt = _load_sp_gt(sp_root, dataset_name, modality='primary')

    pseudo_labels = _load_or_generate_pseudo_labels(
        args, dataset_name, data, aux_data, attri_descriptions, SP_gt)

    data = data.astype(np.float32)
    gt = gt.astype(np.int32)
    class_count = np.max(gt)

    train_ratio = args.curr_train_ratio
    val_ratio = args.curr_val_ratio

    m, n, d = data.shape

    data = np.reshape(data, [m * n, d])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [m, n, d])

    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []
    val_rand_idx = []

    # 样本划分
    if args.samples_type == 'ratio':
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [j for j in range(samplesCount)]
            random.seed(curr_seed)
            rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_ratio).astype('int32'))
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)

        train_data_index = np.array([idx for class_idx in train_rand_idx for idx in class_idx])
        train_data_index = set(train_data_index)
        all_data_index = set(range(len(gt_reshape)))
        background_idx = set(np.where(gt_reshape == 0)[-1])
        test_data_index = all_data_index - train_data_index - background_idx

        val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))
        val_data_index = set(random.sample(list(test_data_index), val_data_count))
        test_data_index = test_data_index - val_data_index

        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)

    # 构建训练/测试标签
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
    Train_Label = np.reshape(train_samples_gt, [m, n])

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
    Test_Label = np.reshape(test_samples_gt, [m, n])

    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
    Val_Label = np.reshape(val_samples_gt, [m, n])

    # 处理 del_class (unseen classes)
    if del_class is not None:
        mask = np.zeros_like(Train_Label)
        mask_g = np.zeros_like(Train_Label)
        for value in del_class:
            mask[Train_Label == value + 1] = 1
            mask_g[Train_Label == value + 1] = 1
        Train_Label_U = Train_Label * (1 - mask_g)
    else:
        Train_Label_U = Train_Label

    # 处理 incremental_classes (持续学习新增类别)
    if incremental_classes is not None:
        # 将增量类别从训练标签中移除，用于模拟新类别出现
        for value in incremental_classes:
            Train_Label_U[Train_Label_U == value + 1] = 0

    # 切分 patch
    Train_Split_Data, Train_Split_GT, Train_Split_PL = SpiltHSI(
        data, Train_Label_U, pseudo_labels, [args.split_height, args.split_width], args.EDGE)
    Test_Split_Data, Test_Split_GT, Test_Split_PL = SpiltHSI(
        data, Test_Label, pseudo_labels, [args.split_height, args.split_width], args.EDGE)

    _, patch_height, patch_width, bands = Train_Split_Data.shape
    patch_height -= args.EDGE * 2
    patch_width -= args.EDGE * 2

    return (
        Train_Split_Data,
        Train_Split_GT,
        Train_Split_PL,
        Test_Split_Data,
        Test_Split_GT,
        Test_Split_PL,
        patch_height,
        patch_width,
        data,
        aux_data,  # 新增返回辅助模态数据
        gt,
        SP_gt,
        pseudo_labels,
        Train_Label,
        Train_Label_U,
        Test_Label,
    )


def read_incremental_data(args, curr_seed, seen_classes, new_classes):
    """
    读取用于增量学习的数据。

    参数:
        args: 配置参数
        curr_seed: 随机种子
        seen_classes: 已见类别列表 (0-based)
        new_classes: 新类别列表 (0-based)

    返回:
        包含已见类和新类数据的元组
    """
    # 首先读取完整数据
    result = read_data(args, curr_seed)

    (Train_Split_Data, Train_Split_GT, Train_Split_PL,
     Test_Split_Data, Test_Split_GT, Test_Split_PL,
     patch_height, patch_width, data, aux_data, gt,
     SP_gt, pseudo_labels, Train_Label, Train_Label_U, Test_Label) = result

    # 划分已见类和新类数据
    m, n = gt.shape

    # 原始标签用于获取新类数据
    orig_gt = np.reshape(gt, [-1])
    train_label_flat = np.reshape(Train_Label, [-1])

    # 新类别训练样本
    new_class_indices = []
    for cls in new_classes:
        cls_indices = np.where(orig_gt == cls + 1)[0]
        # 从这些样本中选择一部分作为新类别的"伪标签"来源
        # 在实际场景中，这些样本可能只有少量标注或无标注
        new_class_indices.extend(cls_indices[:50].tolist())

    new_class_indices = list(set(new_class_indices))

    return {
        'train_data': Train_Split_Data,
        'train_gt': Train_Split_GT,
        'train_pl': Train_Split_PL,
        'test_data': Test_Split_Data,
        'test_gt': Test_Split_GT,
        'test_pl': Test_Split_PL,
        'patch_height': patch_height,
        'patch_width': patch_width,
        'data': data,
        'aux_data': aux_data,
        'gt': gt,
        'sp_gt': SP_gt,
        'pseudo_labels': pseudo_labels,
        'train_label': Train_Label,
        'train_label_u': Train_Label_U,
        'test_label': Test_Label,
        'seen_classes': seen_classes,
        'new_classes': new_classes,
        'nband': Train_Split_Data.shape[-1],
        'nclass': np.max(gt),
    }


def load_teacher_model(checkpoint_path, device):
    """
    加载教师模型用于知识蒸馏。

    参数:
        checkpoint_path: 教师模型 checkpoint 路径
        device: 计算设备

    返回:
        教师模型
    """
    from seg_model import HSI_Seg_Incremental

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 假设 checkpoint 包含模型配置信息
    # 需要根据保存时的配置加载

    return checkpoint.get('model', checkpoint)


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """保存模型 checkpoint。"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
    }, path)