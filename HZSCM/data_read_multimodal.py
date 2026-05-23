"""
早期融合版的数据读取逻辑。

这份文件与单模态版的主要区别在于：
1. 训练输入不是只保留主模态，而是把多个模态在通道维直接拼接
2. 每个模态会先单独标准化，再进行早期融合
3. 其余流程仍然保持和主项目兼容：读数据、读伪标签、划分样本、切 patch
"""


import random
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from sklearn import preprocessing

from util import SpiltHSI
from data_read_singlemodal import (
    DATASET_CLASS_NAMES,
    _load_multimodal_dataset,
    _load_or_generate_pseudo_labels,
    _load_sp_gt,
    _resolve_dataset_name,
)


def _ensure_3d_cube(data):
    """保证输入数据总是 [H, W, C] 形式，便于后续统一处理。"""
    data = np.asarray(data)
    if data.ndim == 2:
        return data[:, :, None]
    if data.ndim == 3:
        return data
    raise ValueError(f'Expected a 2D or 3D modality array, but got shape {data.shape}')


def _standardize_cube(data):
    """对一个模态做按通道标准化。"""
    data = _ensure_3d_cube(data).astype(np.float32)
    h, w, c = data.shape
    reshaped = data.reshape(h * w, c)
    scaler = preprocessing.StandardScaler()
    reshaped = scaler.fit_transform(reshaped)
    return reshaped.reshape(h, w, c).astype(np.float32)


def _fuse_modalities(primary_data, aux_data):
    """把两个模态直接在通道维拼接，形成早期融合输入。"""
    primary_data = _ensure_3d_cube(primary_data)
    if aux_data is None:
        return primary_data.astype(np.float32)

    aux_data = _ensure_3d_cube(aux_data)
    if primary_data.shape[:2] != aux_data.shape[:2]:
        raise ValueError(
            f'Cannot apply early fusion because modality shapes do not align: '
            f'{primary_data.shape[:2]} vs {aux_data.shape[:2]}'
        )
    return np.concatenate([primary_data, aux_data], axis=2).astype(np.float32)


def _load_legacy_dataset(dataset_name, data_root):
    """兼容项目原来已经支持的历史数据集。"""
    legacy_root = Path(data_root)
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
        attri_descriptions = [
            "road", "grass", "gravel", "trees", "metal sheet roof",
            "soil", "bitumen", "shadow"
        ]
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

    return data, None, gt, attri_descriptions


def read_data(args, curr_seed):
    """
    读取数据并构造早期融合训练输入。

    关键区别是：
    - 主模态和辅助模态都会先标准化
    - 然后在通道维拼接成 fused_data
    - 后面的 patch 切分、训练和测试都基于 fused_data

    关键变量：
    - `data`：主模态整图，形状通常为 `[H, W, C1]`
    - `aux_data`：第二模态整图，形状通常为 `[H, W, C2]`
    - `fused_data`：把两个模态沿通道维拼接后的训练输入
    - `SP_gt`：主模态对应的超像素标签图，后续伪标签修正会依赖它
    - `pseudo_labels`：整图尺度的伪标签概率图
    - `Train_Label / Test_Label`：整图尺度的监督标签掩码
    """
    dataset_name = _resolve_dataset_name(args.dataset)
    multimodal_dataset = _load_multimodal_dataset(dataset_name, Path(args.data_root))
    # 优先读取整理好的双模态数据；没有的话再回退到历史数据集分支。
    if multimodal_dataset is not None:
        data, aux_data, gt, attri_descriptions = multimodal_dataset
    else:
        data, aux_data, gt, attri_descriptions = _load_legacy_dataset(dataset_name, args.data_root)

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

    # 先分别标准化各模态，再做通道拼接，避免某个模态量纲过大主导训练。
    fused_data = _standardize_cube(data)
    if aux_data is not None:
        fused_data = _fuse_modalities(fused_data, _standardize_cube(aux_data))

    gt = gt.astype(np.int32)
    class_count = np.max(gt)
    # 这里保留两种采样模式共用的变量名：
    # - `train_samples_per_class`：same_num 模式下，每类固定采样数
    # - `train_ratio`：ratio 模式下，每类按比例采样
    # - `val_samples / val_ratio`：验证集采样数或采样比例
    train_samples_per_class = args.curr_train_ratio
    val_samples = args.curr_val_ratio
    val_ratio = args.curr_val_ratio
    train_ratio = args.curr_train_ratio

    m, n, _ = fused_data.shape
    # `gt_reshape`：把二维标签图拉平成一维，方便按类别找样本索引。
    gt_reshape = np.reshape(gt, [-1])
    # `train_rand_idx`：按类别保存的训练样本索引列表。
    train_rand_idx = []
    # `val_rand_idx`：原始代码预留变量，目前没有实际使用。
    val_rand_idx = []

    # 构造训练 / 验证 / 测试划分。
    if args.samples_type == 'ratio':
        for i in range(class_count):
            # `idx`：第 i 类在整幅图中的所有像素位置索引。
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [j for j in range(samplesCount)]
            random.seed(curr_seed)
            rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_ratio).astype('int32'))
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)
        # 每个类别抽到的样本数可能不同，这里保持 list 结构再展开，
        # 避免新版 NumPy 因不规则数组直接报错。
        # `train_data_index`：把“按类保存”的训练索引展平成一维数组。
        train_data_index = np.array([idx for class_idx in train_rand_idx for idx in class_idx])

        # `all_data_index`：所有像素位置
        # `background_idx`：背景位置（标签 0）
        # `test_data_index`：非训练、非背景的样本全部进入测试集
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

    # 下面三张标签图是整图尺度的监督掩码：
    # - Train_Label：只有训练位置保留类别编号
    # - Test_Label：只有测试位置保留类别编号
    # - Val_Label：只有验证位置保留类别编号
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

    # 把融合后的整图切成 patch，后续网络直接读取这些 patch。
    Train_Split_Data, Train_Split_GT, Train_Split_PL = SpiltHSI(
        fused_data, Train_Label, pseudo_labels, [args.split_height, args.split_width], args.EDGE
    )
    Test_Split_Data, Test_Split_GT, Test_Split_PL = SpiltHSI(
        fused_data, Test_Label, pseudo_labels, [args.split_height, args.split_width], args.EDGE
    )
    _, patch_height, patch_width, _ = Train_Split_Data.shape
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
        fused_data,
        gt,
        SP_gt,
        pseudo_labels,
        Train_Label,
        Test_Label,
    )
