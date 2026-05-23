# -*- coding: utf-8 -*-
"""
单模态训练版本的数据读取逻辑。

这份文件的特点是：
1. 伪标签可以由单模态或多模态图像生成
2. 但真正送入网络训练的输入，只使用主模态数据
3. 同时负责数据集读取、伪标签读取/生成、数据划分和 patch 切分
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
    """把任意模态数据转成 3 通道图像，供 CLIP 或分割脚本使用。"""
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError(f'expected a 2D or 3D array, but got shape {image.shape}')

    image = image.astype(np.float32)
    channel_count = image.shape[2]

    # 单通道复制成三通道，双通道补一个均值通道，多通道则抽 3 个波段。
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

    # 统一拉伸到 0~255，保证视觉模型输入稳定。
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
    """从 .mat 文件读取数组，优先使用给定键名。"""
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
    """读取当前项目已经整理好的双模态数据集。"""
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
    """
    读取某个模态对应的超像素标签图。

    新的命名约定优先使用：
    - `<dataset>_primary_SPGT_*.mat`
    - `<dataset>_secondary_SPGT_*.mat`

    为了兼容旧实验，主模态仍允许回退到历史命名 `<dataset>_SPGT_*.mat`。
    """
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
    raise FileNotFoundError(
        f'Cannot find a superpixel file under {sp_root} for dataset {dataset_name}. '
        f'Expected a file like {expected}.'
    )


def _load_sam_list(sp_root, dataset_name, modality='primary'):
    """
    读取某个模态对应的 SAM 自动分割 mask 列表。

    新的命名约定优先使用：
    - `<dataset>_primary_list.pkl`
    - `<dataset>_secondary_list.pkl`

    为了兼容旧实验，主模态仍允许回退到历史命名 `<dataset>_list.pkl`。
    """
    for prefix in _candidate_dataset_prefixes(dataset_name):
        candidates = [sp_root / f'{prefix}_{modality}_list.pkl']
        if modality == 'primary':
            candidates.append(sp_root / f'{prefix}_list.pkl')
        for candidate in candidates:
            if candidate.exists():
                with open(candidate, 'rb') as file:
                    return pickle.load(file)
    expected = f'{dataset_name}_{modality}_list.pkl'
    if modality == 'primary':
        expected += f' or legacy {dataset_name}_list.pkl'
    raise FileNotFoundError(
        f'Cannot find a SAM segmentation list under {sp_root} for dataset {dataset_name}. '
        f'Expected a file like {expected}.'
    )


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
    """构造项目约定的伪标签缓存文件名。"""
    suffix = clip_model_name[args.multi_clip]
    if multimodal:
        filename = f'{dataset_name}_margin{args.margin}_usesam{args.use_sam}{suffix}_permodality_avg_pseudo_labels.mat'
    else:
        filename = f'{dataset_name}_margin{args.margin}_usesam{args.use_sam}{suffix}_pseudo_labels.mat'
    return sp_root / filename


def _load_or_generate_pseudo_labels(args, dataset_name, data, aux_data, attri_descriptions, sp_gt):
    """
    优先从磁盘读取伪标签；如果配置为 generate，则在线生成并写回缓存。

    关键变量：
    - `use_multimodal`：是否需要同时使用主模态和第二模态来生成伪标签
    - `cache_path`：当前配置对应的缓存文件路径
    - `modality_images`：送给 CLIP 的 RGB 图列表
    - `modality_sp_gts`：每个模态各自对应的 ERS 超像素图
    - `modality_sam_lists`：每个模态各自对应的 SAM mask 列表
    - `clip_outputs`：不同 CLIP 权重生成的概率图，最终还会继续平均
    """
    sp_root = Path(args.sp_root)
    use_multimodal = bool(args.use_multimodal_pseudo) and aux_data is not None

    cache_path = _pseudo_label_cache_path(sp_root, dataset_name, args, use_multimodal)
    single_cache_path = _pseudo_label_cache_path(sp_root, dataset_name, args, False)

    # 训练阶段默认先用缓存，避免每次都重复跑 CLIP。
    if args.pseudo_label_mode == 'load':
        if cache_path.exists():
            return sio.loadmat(cache_path)['pseudo_labels']
        if not use_multimodal and single_cache_path.exists():
            return sio.loadmat(single_cache_path)['pseudo_labels']
        raise FileNotFoundError(
            f'Cannot find pseudo labels for {dataset_name}. '
            f'Expected {cache_path}. '
            f'Set --pseudo_label_mode generate to create it.'
        )

    if len(attri_descriptions) == 0:
        raise ValueError(f'No class descriptions are configured for dataset {dataset_name}')
    if CLIPModel is None or CLIPProcessor is None:
        raise ImportError(
            'transformers is required to generate pseudo labels. '
            'Install transformers first or switch --pseudo_label_mode to load.'
        )

    # 如果是双模态伪标签，这里会准备两个模态图像并分别送入 CLIP。
    modality_images = _build_modality_images(dataset_name, data, aux_data, use_multimodal)
    # `sp_gt` 已经是主模态的超像素图；第二模态的超像素图需要额外读取。
    modality_sp_gts = [sp_gt]
    if use_multimodal:
        modality_sp_gts.append(_load_sp_gt(sp_root, dataset_name, modality='secondary'))

    # 只有启用 SAM 时，才需要读取每个模态对应的 SAM 掩码列表。
    modality_sam_lists = None
    if args.use_sam:
        modality_sam_lists = [_load_sam_list(sp_root, dataset_name, modality='primary')]
        if use_multimodal:
            modality_sam_lists.append(_load_sam_list(sp_root, dataset_name, modality='secondary'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_outputs = []

    with torch.no_grad():
        # 支持一个或两个 CLIP 模型，最后统一在概率层做平均。
        for clip_dir in _get_clip_dirs(args):
            clip_model = CLIPModel.from_pretrained(clip_dir).to(device)
            processor = CLIPProcessor.from_pretrained(clip_dir)
            # use_sam=1 时，伪标签生成会基于每个模态自己的 SAM + ERS 融合区域。
            if args.use_sam and use_multimodal:
                pseudo_probs = get_multimodal_pseudo_labels_using_individual_SAM_and_ERS(
                    modality_sp_gts,
                    modality_sam_lists,
                    modality_images,
                    clip_model,
                    processor,
                    attri_descriptions,
                    margin=args.margin,
                )
            elif args.use_sam:
                pseudo_probs = get_pseudo_labels_using_SAM_and_ERS(
                    modality_sp_gts[0],
                    modality_sam_lists[0],
                    modality_images[0],
                    clip_model,
                    processor,
                    attri_descriptions,
                    margin=args.margin,
                )
            elif use_multimodal:
                pseudo_probs = get_multimodal_pseudo_labels_with_individual_segments(
                    modality_sp_gts,
                    modality_images,
                    clip_model,
                    processor,
                    attri_descriptions,
                    margin=args.margin,
                )
            else:
                pseudo_probs = get_pseudo_labels(
                    modality_sp_gts[0],
                    modality_images[0],
                    clip_model,
                    processor,
                    attri_descriptions,
                    margin=args.margin,
                )
            clip_outputs.append(pseudo_probs.cpu())
            del clip_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 多个 CLIP 输出再次平均，得到最终伪标签概率图。
    # 多个 CLIP 输出再次平均，得到最终整图伪标签概率。
    pseudo_labels = torch.stack(clip_outputs, dim=0).mean(dim=0).numpy().astype(np.float32)
    save_path = cache_path if use_multimodal else single_cache_path
    sio.savemat(save_path, {'pseudo_labels': pseudo_labels})
    return pseudo_labels


def read_data(args, curr_seed):
    """
    读取当前数据集，并返回训练主流程需要的全部内容。

    这里会依次完成：
    1. 读取原始数据和类别文本描述
    2. 读取超像素图与伪标签
    3. 标准化主模态数据
    4. 按类别划分训练 / 验证 / 测试样本
    5. 把整图切成 patch

    关键变量：
    - `data`：主模态整图输入
    - `aux_data`：第二模态整图，仅伪标签生成时可能使用
    - `gt`：整图真实标签，标签编码为 `0..C` 或 `1..C`
    - `SP_gt`：主模态超像素图
    - `pseudo_labels`：整图伪标签概率
    - `Train_Split_* / Test_Split_*`：切成 patch 后的数据和标签
    """
    dataset_name = _resolve_dataset_name(args.dataset)
    multimodal_dataset = _load_multimodal_dataset(dataset_name, Path(args.data_root))
    aux_data = None

    # 先尝试读取项目整理好的双模态数据集；否则回退到历史单模态数据集分支。
    if multimodal_dataset is not None:
        data, aux_data, gt, attri_descriptions = multimodal_dataset
    else:
        legacy_root = Path(args.data_root)
        if not legacy_root.exists():
            legacy_root = Path('F:/HSIdata')

        if dataset_name == 'QUH_Pingan':
            data_file = legacy_root / 'Matlab format' / 'QUH-Pingan' / 'QUH-Pingan.mat'
            gt_file = legacy_root / 'Matlab format' / 'QUH-Pingan' / 'QUH-Pingan_GT.mat'
            data_mat = sio.loadmat(data_file)
            gt_mat = sio.loadmat(gt_file)
            data = data_mat['Haigang']
            gt = gt_mat['HaigangGT']
            attri_descriptions = [
                "ship", "seawater", "trees", "concrete structure building",
                "floating pier", "brick houses", "steel houses",
                "wharf construction land", "car", "road"
            ]
        elif dataset_name == 'Pavia':
            data_file = legacy_root / 'Pavia' / 'Pavia.mat'
            gt_file = legacy_root / 'Pavia' / 'Pavia_groundtruth.mat'
            data_mat = sio.loadmat(data_file)
            gt_mat = sio.loadmat(gt_file)
            data = data_mat['paviaU']
            gt = gt_mat['paviaU_gt']
            attri_descriptions = DATASET_CLASS_NAMES['Pavia']
        elif dataset_name == 'PaviaR':
            data_file = legacy_root / 'Pavia' / 'Pavia.mat'
            gt_file = legacy_root / 'Pavia' / 'Pavia_groundtruth.mat'
            data_mat = sio.loadmat(data_file)
            gt_mat = sio.loadmat(gt_file)
            data = data_mat['paviaU']
            gt_o = gt_mat['paviaU_gt']
            gt = np.copy(gt_o)
            label_map = [0, 1, 2, 3, 4, 5, 6, 0, 7]
            attri_descriptions = DATASET_CLASS_NAMES['PaviaR']
            for i in range(len(label_map)):
                gt[gt_o == i + 1] = label_map[i] + 1
        elif dataset_name == 'QUH_Qingyun':
            image_file = legacy_root / 'Matlab format' / 'QUH-Qingyun' / 'QUH-Qingyun.mat'
            label_file = legacy_root / 'Matlab format' / 'QUH-Qingyun' / 'QUH-Qingyun_GT.mat'
            image_data = sio.loadmat(image_file)
            label_data = sio.loadmat(label_file)
            data = image_data['Chengqu']
            gt = label_data['ChengquGT']
            attri_descriptions = [
                "Trees", "building", "car", "sheet metal",
                "plastic playground", "asphalt road"
            ]
        elif dataset_name == 'CASI':
            data_file = legacy_root / 'Houston' / 'CASI.mat'
            gt_file = legacy_root / 'Houston' / 'CASI_gnd_flag.mat'
            gt_mat = sio.loadmat(gt_file)
            data_mat = sio.loadmat(data_file)
            data = data_mat['CASI']
            gt = gt_mat['gnd_flag']
            attri_descriptions = DATASET_CLASS_NAMES['Houston']
        elif dataset_name == 'CASIR':
            data_file = legacy_root / 'Houston' / 'CASI.mat'
            gt_file = legacy_root / 'Houston' / 'CASI_gnd_flag.mat'
            gt_mat = sio.loadmat(gt_file)
            data_mat = sio.loadmat(data_file)
            data = data_mat['CASI']
            gt_o = gt_mat['gnd_flag']
            gt = np.copy(gt_o)
            label_map = [0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11]
            attri_descriptions = [
                "grass", "synthetic grass", "trees", "soil", "water",
                "buildings", "road", "highway", "railway", "parking lot",
                "tennis court", "running track"
            ]
            for i in range(len(label_map)):
                gt[gt_o == i + 1] = label_map[i] + 1
        elif dataset_name == 'CASI2018':
            image_file = legacy_root / 'HoustonU' / 'HoustonU.mat'
            label_file = legacy_root / 'HoustonU' / 'HoustonU_gt.mat'
            image_data = h5py.File(image_file, 'r')
            label_data = h5py.File(label_file, 'r')
            data = np.transpose(np.array(image_data['houstonU']), (2, 1, 0))
            gt_o = np.transpose(np.array(label_data['houstonU_gt']), (1, 0))
            gt = np.copy(gt_o)
            label_map = [0, 0, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12]
            attri_descriptions = [
                "grass", "synthetic grass", "trees", "soil", "water", "buildings",
                "roads", "highways", "railways", "parking lots", "cars",
                "trains", "stadium seats"
            ]
            for i in range(len(label_map)):
                gt[gt_o == i + 1] = label_map[i] + 1
        else:
            raise Exception('dataset does not find')

    # 读取超像素标签图和伪标签概率图。
    sp_root = Path(args.sp_root)
    # 后续伪标签修正仍以主模态的超像素图作为统一基准。
    SP_gt = _load_sp_gt(sp_root, dataset_name, modality='primary')
    pseudo_labels = _load_or_generate_pseudo_labels(
        args,
        dataset_name,
        data,
        aux_data,
        attri_descriptions,
        SP_gt,
    )

    # 单模态版本训练时只使用主模态，因此这里只标准化主模态。
    data = data.astype(np.float32)
    gt = gt.astype(np.int32)
    class_count = np.max(gt)

    # 采样相关变量名与多模态版本保持一致，方便两个文件对照阅读。
    train_samples_per_class = args.curr_train_ratio
    val_samples = args.curr_val_ratio
    val_ratio = args.curr_val_ratio
    train_ratio = args.curr_train_ratio

    m, n, d = data.shape

    data = np.reshape(data, [m * n, d])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [m, n, d])

    # `gt_reshape`：把二维标签图拉平，便于按类别找索引。
    gt_reshape = np.reshape(gt, [-1])
    # `train_rand_idx`：按类别保存的训练像素位置。
    train_rand_idx = []
    # `val_rand_idx`：原始代码遗留变量，目前未实际使用。
    val_rand_idx = []

    # 样本划分：按类别抽取训练样本，其余样本进入测试/验证集合。
    if args.samples_type == 'ratio':
        for i in range(class_count):
            # `idx`：当前类别所有像素在拉平数组中的位置。
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [j for j in range(samplesCount)]
            random.seed(curr_seed)
            rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_ratio).astype('int32'))
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)
        # 每个类别抽到的样本数可能不同，这里保持 list 结构再展开，
        # 避免新版 NumPy 因不规则数组直接报错。
        # 把“按类保存”的索引展平成一维训练索引数组。
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

    if args.samples_type == 'same_num':
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = int(train_samples_per_class)
            rand_list = [j for j in range(samplesCount)]
            if real_train_samples_per_class > samplesCount:
                real_train_samples_per_class = int(train_samples_per_class / 2)
            random.seed(curr_seed)
            rand_idx = random.sample(rand_list, real_train_samples_per_class)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)
        train_data_index = np.array([idx for class_idx in train_rand_idx for idx in class_idx])

        train_data_index = set(train_data_index)
        all_data_index = set(range(len(gt_reshape)))
        background_idx = set(np.where(gt_reshape == 0)[-1])
        test_data_index = all_data_index - train_data_index - background_idx

        val_data_count = int(val_samples)
        val_data_index = set(random.sample(list(test_data_index), val_data_count))
        test_data_index = test_data_index - val_data_index

        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)

    # 下面三张图都是“只在指定位置保留标签，其余位置置零”的监督掩码。
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

    # 最后把整图切成 patch，后续 DataLoader 会直接读取这些 patch。
    Train_Split_Data, Train_Split_GT, Train_Split_PL = SpiltHSI(
        data, Train_Label, pseudo_labels, [args.split_height, args.split_width], args.EDGE
    )
    Test_Split_Data, Test_Split_GT, Test_Split_PL = SpiltHSI(
        data, Test_Label, pseudo_labels, [args.split_height, args.split_width], args.EDGE
    )
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
        gt,
        SP_gt,
        pseudo_labels,
        Train_Label,
        Test_Label,
    )
