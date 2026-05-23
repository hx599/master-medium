# -*- coding: utf-8 -*-
"""
工具函数集合 - 持续学习版本。

基于 HZSCM util.py 扩展，新增：
1. 模态适配器投影
2. 增量学习场景的伪标签生成
3. 知识蒸馏相关工具
4. 增量数据加载辅助函数

变量约定 (沿用 HZSCM):
- `data / image / HSI_image`: 整图输入 [H, W, C]
- `gt / target / Train_Label`: 真实标签图 [H, W]
- `sp_gt / SP_label / ERS_label`: 超像素标签图 [H, W]
- `pseudo_probs / clip_probs`: 伪标签概率图 [H, W, num_classes]
"""

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from operator import truediv
from sklearn import metrics, neighbors
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sklearn.cluster as clu
import pickle

try:
    import spectral as spy
except ImportError:
    spy = None


# ==================== 沿用 HZSCM 的核心函数 ====================

def get_pseudo_labels(SP_label, image, model, processor, label_name, margin=0, resize=True):
    """基于超像素区域，用 CLIP 为整幅图生成伪标签概率图。"""
    device = model.device
    num_SP = np.max(SP_label + 1)
    image_probs = torch.zeros([image.shape[0], image.shape[1], len(label_name)])

    for i in np.unique(SP_label):
        i_row, i_col = np.where(SP_label == i)
        row_max, row_min = np.max(i_row), np.min(i_row)
        col_max, col_min = np.max(i_col), np.min(i_col)

        row_length = int(margin * (row_max - row_min))
        col_length = int(margin * (col_max - col_min))

        if row_max + row_length >= image.shape[0] or row_min - row_length <= 0 or \
           col_min - col_length <= 0 or col_max + row_length >= image.shape[1]:
            i_image = Image.fromarray(image[row_min:row_max+1, col_min:col_max+1])
        else:
            i_image = Image.fromarray(image[row_min - row_length:row_max + row_length + 1,
                                           col_min - col_length:col_max + col_length + 1])

        if resize:
            resized_image = i_image.resize((224, 224), Image.BICUBIC)
            inputs = processor(text=[f"a photo of {l}" for l in label_name],
                             images=resized_image, return_tensors="pt", padding=True)
        else:
            inputs = processor(text=[f"a photo of {l}" for l in label_name],
                             images=i_image, return_tensors="pt", padding=True)

        for name, tensor in inputs.items():
            inputs[name] = tensor.to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        image_probs[i_row, i_col] = probs.cpu()

    return image_probs


def fuse_SAM_ERS(ERS_label, SAM_list):
    """把 SAM 的分割结果覆盖到 ERS 上。"""
    SP_num = 0
    new_SAM_label = np.ones_like(ERS_label) * -100
    new_ERS_label = np.ones_like(ERS_label) * -100

    for i_list in SAM_list:
        i_SP = i_list['segmentation']
        new_SAM_label[i_SP] = SP_num
        SP_num = SP_num + 1

    residual_ERS_label = ERS_label[new_SAM_label <= -1]
    residual_ERS_index = np.unique(residual_ERS_label)

    for i in residual_ERS_index:
        new_ERS_label[ERS_label == i] = i

    return new_SAM_label, new_ERS_label


def get_pseudo_labels_using_SAM_and_ERS(ERS_label, SAM_list, image, model, processor, label_name, margin=0, resize=True):
    """先融合 SAM 与 ERS，再基于融合结果生成伪标签。"""
    new_SAM_label, new_ERS_label = fuse_SAM_ERS(ERS_label, SAM_list)
    SAM_probs = get_pseudo_labels(new_SAM_label, image, model, processor, label_name, margin=margin, resize=resize)
    ERS_probs = get_pseudo_labels(new_ERS_label, image, model, processor, label_name, margin=margin, resize=resize)

    image_probs = torch.zeros_like(SAM_probs)
    image_probs[new_SAM_label > -1] = SAM_probs[new_SAM_label > -1]
    image_probs[new_SAM_label <= -1] = ERS_probs[new_SAM_label <= -1]

    return image_probs


def average_pseudo_probs(prob_list):
    """对多个概率图做平均。"""
    if len(prob_list) == 0:
        raise ValueError('prob_list should not be empty')
    tensors = []
    for probs in prob_list:
        if torch.is_tensor(probs):
            tensors.append(probs.float())
        else:
            tensors.append(torch.as_tensor(probs, dtype=torch.float32))
    return torch.stack(tensors, dim=0).mean(dim=0)


def get_multimodal_pseudo_labels_with_individual_segments(SP_labels, images, model, processor, label_name, margin=0, resize=True):
    """多模态版本：每个模态使用自己的超像素分割结果生成伪标签，再在概率层平均。"""
    if len(SP_labels) != len(images):
        raise ValueError('SP_labels and images should have the same length')
    prob_list = [
        get_pseudo_labels(sp_label, image, model, processor, label_name, margin=margin, resize=resize)
        for sp_label, image in zip(SP_labels, images)
    ]
    return average_pseudo_probs(prob_list)


def get_multimodal_pseudo_labels_using_individual_SAM_and_ERS(ERS_labels, SAM_lists, images, model, processor, label_name, margin=0, resize=True):
    """多模态版本：每个模态各自使用自己的 ERS 与 SAM 分割结果。"""
    if not (len(ERS_labels) == len(SAM_lists) == len(images)):
        raise ValueError('ERS_labels, SAM_lists, and images should have the same length')
    prob_list = [
        get_pseudo_labels_using_SAM_and_ERS(ers_label, sam_list, image, model, processor, label_name, margin=margin, resize=resize)
        for ers_label, sam_list, image in zip(ERS_labels, SAM_lists, images)
    ]
    return average_pseudo_probs(prob_list)


# ==================== 新增：模态适配器相关 ====================

class ModalityProjector:
    """
    模态适配器：将非光学模态(SAR/LiDAR)投影到 CLIP 视觉编码器兼容空间。

    在 CLIP 推理之前对非光学模态数据应用轻量级变换，
    改善其与 CLIP 语义空间的对齐。
    """

    def __init__(self, input_channels, output_dim=768, hidden_dim=256):
        """
        参数:
            input_channels: 输入模态的通道数
            output_dim: 输出维度 (CLIP 视觉编码器期望的维度)
            hidden_dim: 中间层维度
        """
        self.input_channels = input_channels
        self.output_dim = output_dim

        # 轻量级 MLP 投影
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def project(self, features):
        """
        投影特征到 CLIP 空间。

        参数:
            features: 输入特征 [N, C]

        返回:
            投影后的特征 [N, output_dim]
        """
        return self.mlp(features)

    def to_rgb_like(self, data, band_indices=None):
        """
        将任意模态数据转换为类似 RGB 的 3 通道表示 (作为备用方案)。

        参数:
            data: 原始数据 [H, W, C]
            band_indices: 用于 RGB 的波段索引

        返回:
            rgb_like: [H, W, 3]
        """
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[:, :, None]

        C = data.shape[2]
        if band_indices is None:
            # 默认取前3个或等间隔3个波段
            if C >= 3:
                band_indices = [0, C // 2, C - 1]
            else:
                band_indices = list(range(C))
                while len(band_indices) < 3:
                    band_indices.append(band_indices[-1])

        rgb_like = data[:, :, band_indices[:3]]

        # 归一化到 0-255
        rgb_like = rgb_like - np.min(rgb_like, axis=(0, 1), keepdims=True)
        rgb_max = np.max(rgb_like, axis=(0, 1), keepdims=True)
        rgb_max[rgb_max < 1e-6] = 1.0
        rgb_like = np.clip(rgb_like / rgb_max * 255.0, 0, 255).astype(np.uint8)

        # 扩展到3通道
        while rgb_like.shape[2] < 3:
            rgb_like = np.concatenate([rgb_like, rgb_like[:, :, -1:]], axis=2)

        return rgb_like


# ==================== 新增：增量学习辅助函数 ====================

def prepare_incremental_data(original_data, original_gt, seen_classes, new_classes):
    """
    将原始数据划分为已见类和新类样本。

    参数:
        original_data: 原始数据 [H, W, C]
        original_gt: 原始标签 [H, W]，0为背景，1..N为类别
        seen_classes: 已见类别列表 (0-based)
        new_classes: 新类别列表 (0-based)

    返回:
        seen_mask: 已见类别的掩码 (bool)
        new_mask: 新类别的掩码 (bool)
    """
    seen_mask = np.isin(original_gt, [c + 1 for c in seen_classes])
    new_mask = np.isin(original_gt, [c + 1 for c in new_classes])

    return seen_mask, new_mask


def compute_class_incremental_metrics(pred, gt, seen_classes, new_classes, n_total_classes):
    """
    计算增量学习的评价指标。

    返回:
        seen_accuracy: 已见类精度
        new_accuracy: 新类精度
        overall_accuracy: 总体精度
        harmonic_mean: 调和均值
    """
    # 计算每类精度
    ac = np.zeros(n_total_classes)
    cnt = np.zeros(n_total_classes)

    for i in range(n_total_classes):
        mask = (gt == i + 1)
        cnt[i] = np.sum(mask)
        if cnt[i] > 0:
            ac[i] = np.sum((pred == gt) & mask) / cnt[i]

    # 已见类和新类
    seen_mask = np.array([i in seen_classes for i in range(n_total_classes)])
    new_mask = np.array([i in new_classes for i in range(n_total_classes)])

    seen_accuracy = np.mean(ac[seen_mask]) if np.sum(cnt[seen_mask]) > 0 else 0
    new_accuracy = np.mean(ac[new_mask]) if np.sum(cnt[new_mask]) > 0 else 0

    # 总体精度
    valid_mask = gt > 0
    overall_accuracy = np.sum((pred == gt) & valid_mask) / np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0

    # 调和均值
    if seen_accuracy + new_accuracy > 0:
        harmonic_mean = 2 * seen_accuracy * new_accuracy / (seen_accuracy + new_accuracy)
    else:
        harmonic_mean = 0

    return seen_accuracy, new_accuracy, overall_accuracy, harmonic_mean


def generate_incremental_pseudo_labels(model, data, aux_data, class_names, device):
    """
    为增量学习生成伪标签。

    参数:
        model: 当前模型
        data: 主模态数据 [H, W, C]
        aux_data: 辅助模态数据 [H, W, C_aux]
        class_names: 类别名称列表
        device: 计算设备

    返回:
        pseudo_probs: 伪标签概率图 [H, W, n_classes]
    """
    model.eval()
    H, W, C = data.shape

    # 转换为 tensor 并添加 batch 维
    data_tensor = torch.from_numpy(data).float().permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        if aux_data is not None:
            aux_tensor = torch.from_numpy(aux_data).float().permute(2, 0, 1).unsqueeze(0).to(device)
            _, logits = model.test(data_tensor, aux_tensor)
        else:
            _, logits = model.test(data_tensor)

    # 转换为概率
    probs = F.softmax(logits, dim=-1)

    return probs.squeeze(0).cpu().numpy()


def select_replay_samples(gt, samples_per_class=5, seed=42):
    """
    从每个类别中选择代表性样本用于经验回放。

    参数:
        gt: 标签图 [H, W]
        samples_per_class: 每个类别选择的样本数
        seed: 随机种子

    返回:
        replay_indices: 用于回放的像素索引列表
    """
    np.random.seed(seed)
    replay_indices = []

    classes = np.unique(gt)
    classes = classes[classes > 0]  # 排除背景

    for cls in classes:
        cls_indices = np.where(gt == cls)
        cls_indices = list(zip(cls_indices[0], cls_indices[1]))

        if len(cls_indices) > samples_per_class:
            selected = np.random.choice(len(cls_indices), samples_per_class, replace=False)
            selected_indices = [cls_indices[i] for i in selected]
        else:
            selected_indices = cls_indices

        replay_indices.extend(selected_indices)

    return replay_indices


# ==================== 沿用 HZSCM 的其他函数 ====================

def ClassificationAccuracy(output, target, classcount):
    """计算每类精度、OA 和 AA。"""
    m, n = output.shape

    correct_perclass = np.zeros([classcount])
    count_perclass = np.zeros([classcount])
    count = 0
    aa = 0

    for i in range(m):
        for j in range(n):
            if target[i, j] != 0:
                count = count + 1
                count_perclass[int(target[i, j] - 1)] += 1
                if output[i, j] == target[i, j]:
                    aa = aa + 1
                    correct_perclass[int(target[i, j] - 1)] += 1

    test_AC_list = correct_perclass / count_perclass
    test_AA = np.average(test_AC_list)
    test_OA = aa / count

    return test_AC_list, test_OA, test_AA, aa, count


def Kappa(output, target, classcount):
    """计算 Cohen's kappa 和混淆矩阵。"""
    test_pre_label_list = []
    test_real_label_list = []

    m, n = output.shape
    for ii in range(m):
        for jj in range(n):
            if target[ii][jj] != 0:
                test_pre_label_list.append(output[ii][jj])
                test_real_label_list.append(target[ii][jj])

    test_pre_label_list = np.array(test_pre_label_list)
    test_real_label_list = np.array(test_real_label_list)

    kappa = metrics.cohen_kappa_score(
        test_pre_label_list.astype(np.int16),
        test_real_label_list.astype(np.int16)
    )
    cm = metrics.confusion_matrix(
        test_real_label_list.astype(np.int16),
        test_pre_label_list.astype(np.int16)
    )

    return kappa, cm


def SpiltHSI(data, gt, sp_gt, split_size, edge):
    """把整幅图切成多个 patch。"""
    e = edge
    split_height = split_size[0]
    split_width = split_size[1]
    m, n, d = data.shape

    GT = gt
    SPGT = sp_gt

    # 补0变为可整除
    if m % split_height != 0 or n % split_width != 0:
        data = np.pad(data, [[0, split_height - m % split_height],
                            [0, split_width - n % split_width], [0, 0]], mode='constant')
        GT = np.pad(GT, [[0, split_height - m % split_height],
                        [0, split_width - n % split_width]], mode='constant')
        SPGT = np.pad(SPGT, [[0, split_height - m % split_height],
                            [0, split_width - n % split_width], [0, 0]], mode='constant')

    m_height = int(data.shape[0] / split_height)
    m_width = int(data.shape[1] / split_width)

    pad_data = np.pad(data, [[e, e], [e, e], [0, 0]], mode="constant")
    pad_GT = np.pad(GT, [[e, e], [e, e]], mode="constant")
    pad_SPGT = np.pad(SPGT, [[e, e], [e, e], [0, 0]], mode="constant")

    final_data = []
    final_gt = []
    final_spgt = []

    for i in range(split_height):
        for j in range(split_width):
            temp1 = pad_data[i * m_height:i * m_height + m_height + 2 * e,
                           j * m_width:j * m_width + m_width + 2 * e, :]
            temp2 = pad_GT[i * m_height:i * m_height + m_height + 2 * e,
                          j * m_width:j * m_width + m_width + 2 * e]
            temp3 = pad_SPGT[i * m_height:i * m_height + m_height + 2 * e,
                            j * m_width:j * m_width + m_width + 2 * e, :]
            final_data.append(temp1)
            final_gt.append(temp2)
            final_spgt.append(temp3)

    final_data = np.array(final_data)
    final_gt = np.array(final_gt)
    final_spgt = np.array(final_spgt)

    return final_data, final_gt, final_spgt


def PatchStack(OutPut, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count):
    """把 patch 级预测重新拼回整图。"""
    HSI_stack = np.zeros([split_height * patch_height, split_width * patch_width, class_count], dtype=np.float32)

    for i in range(split_height):
        for j in range(split_width):
            if EDGE == 0:
                HSI_stack[i * patch_height:(i + 1) * patch_height,
                          j * patch_width:(j + 1) * patch_width, :] = OutPut[i * split_width + j][EDGE:, EDGE:, :]
            else:
                HSI_stack[i * patch_height:(i + 1) * patch_height,
                          j * patch_width:(j + 1) * patch_width, :] = OutPut[i * split_width + j][EDGE:-EDGE, EDGE:-EDGE, :]

    HSI_stack = np.argmax(HSI_stack, axis=2)
    HSI_stack = HSI_stack[0:m, 0:n]

    return HSI_stack


def get_USH(ac, test_num, unseen_classes, metric='OA'):
    """计算 seen / unseen / harmonic 指标。"""
    if unseen_classes is None:
        return 0, 0, 0

    test_num = np.array(test_num)
    seen_classes = [i for i in range(len(ac)) if i not in unseen_classes]

    if metric == 'OA':
        S = np.sum(ac[seen_classes] * test_num[seen_classes]) / np.sum(test_num[seen_classes])
        U = np.sum(ac[unseen_classes] * test_num[unseen_classes]) / np.sum(test_num[unseen_classes])
    elif metric == 'AA':
        S = np.mean(ac[seen_classes])
        U = np.mean(ac[unseen_classes])
    else:
        raise Exception('metric should be chosen from [OA, AA]')

    H = 2 * S * U / (S + U) if (S + U) > 0 else 0

    return S, U, H


# ==================== 沿用 HZSCM 的伪标签修正函数 ====================

def local_aggregation(unseen_classes, sp_features, sp_labels, cluster_index, pseudo_probs, threshold, tao):
    """基于局部一致性修正超像素级伪标签。"""
    sim = -metrics.pairwise_distances(sp_features, metric='euclidean') / tao
    pseudo_labels = np.argmax(pseudo_probs, axis=1)
    pseudo_probs_c = np.copy(pseudo_probs)

    ind = sp_labels > -0.5
    pseudo_labels_1 = np.copy(pseudo_labels)

    num_fea = sp_features.shape[0]
    scores = np.ones_like(pseudo_probs) * -100

    for i in range(num_fea):
        j = 0
        while pseudo_labels[i] not in unseen_classes:
            prob = np.exp(sim[i]) / np.sum(np.exp(sim[i]))

            ps_label = pseudo_labels[i]
            cs_label = cluster_index[i]

            spec_neigh_ind = cluster_index == cs_label
            spec_neigh_ind[i] = False

            class_neigh_ind = sp_labels == ps_label
            class_neigh_ind[i] = False
            intersection = spec_neigh_ind & class_neigh_ind

            cs_prob = np.sum(prob[intersection])
            class_prob = np.sum(prob[class_neigh_ind]) + 1e-10

            score = cs_prob / class_prob
            scores[i, j] = score

            if score < threshold:
                j = j + 1
                m = np.copy(pseudo_labels[i])
                pseudo_labels[i] = np.argsort(pseudo_probs_c[i])[-j - 1]
                n = np.copy(pseudo_probs_c[i][pseudo_labels[i]])
                pseudo_probs_c[i][pseudo_labels[i]] = pseudo_probs_c[i][m]
                pseudo_probs_c[i][m] = n
            else:
                break

    return scores, pseudo_labels, pseudo_probs_c


def get_sp_label(HSI_image, sp_gt, train_gt):
    """把像素级数据聚合到超像素级。"""
    sp = np.unique(sp_gt)
    sp_features = []
    sp_labels = []

    for i in sp:
        sp_ind = np.where(sp_gt == i)
        sp_label = train_gt[sp_ind]
        count = np.bincount(sp_label)[1:]
        if np.sum(count) > 0.5:
            sp_label = np.argmax(count)
        else:
            sp_label = -1
        sp_labels.append(sp_label)
        sp_features.append(np.mean(HSI_image[sp_ind], axis=0))

    sp_features = np.array(sp_features)
    sp_labels = np.array(sp_labels)

    return sp_features, sp_labels, sp


def get_cluster_labels(sp_features, sp_labels, n_cluster=40):
    """对超像素特征做谱聚类。"""
    kmeans = clu.SpectralClustering(n_clusters=n_cluster, assign_labels='cluster_qr').fit(sp_features)
    cluster_index = kmeans.labels_

    clusters = []
    for i in np.unique(cluster_index):
        clusters.append(np.where(cluster_index == i)[0])

    p_label = np.copy(sp_labels)
    for i in range(len(clusters)):
        cla = np.zeros(np.max(sp_labels) + 1, dtype=np.int32)
        for j in clusters[i]:
            ind = j
            if sp_labels[ind] >= 0:
                cla[sp_labels[ind]] += 1
        if np.sum(cla) > 0:
            for j in clusters[i]:
                ind = j
                if sp_labels[ind] < 0:
                    p_label[ind] = np.argmax(cla)

    return clusters, cluster_index, p_label


def get_whole_labels(clip_ps_labels, ps_labels, sp_gt):
    """把超像素级标签回填成像素级整图标签。"""
    sp = np.unique(sp_gt)
    whole_labels = np.copy(clip_ps_labels)
    for i in sp:
        sp_ind = np.where(sp_gt == i)
        whole_labels[sp_ind] = ps_labels[i]
    return whole_labels


def correct_pseudo_labels(HSI_image, gt, sp_gt, train_gt, clip_probs, net_ps_labels, unseen_classes,
                          n_neighbours=10, n_clusters=40, tao=1.0, threshold=0.1):
    """伪标签修正总入口。"""
    clip_ps_probs, _, _ = get_sp_label(clip_probs, sp_gt, train_gt.astype(np.int32))
    net_ps_labels = np.array([np.argmax(np.bincount(net_ps_labels[np.where(sp_gt == i)]))
                              for i in np.unique(sp_gt)])

    sp_features, sp_labels, sp_index = get_sp_label(HSI_image, sp_gt, train_gt.astype(np.int32))

    clusters, cluster_index, cluster_labels = get_cluster_labels(sp_features, sp_labels, n_cluster=n_clusters)

    scores, pseudo_labels, pseudo_prob = local_aggregation(
        unseen_classes, sp_features, sp_labels, cluster_index, clip_ps_probs, threshold, tao)

    # 已知监督标签的超像素恢复真值
    ind = sp_labels > -0.5
    pseudo_labels[ind] = sp_labels[ind]
    pseudo_prob[ind] = np.eye(np.max(gt))[sp_labels[ind]]

    whole_pseudo_labels = get_whole_labels(np.argmax(clip_probs, axis=-1), pseudo_labels, sp_gt)
    whole_pseudo_probs = get_whole_labels(clip_probs, pseudo_prob, sp_gt)

    return scores, whole_pseudo_labels, whole_pseudo_probs


# ==================== 学习率调度器 ====================

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr

        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_epochs)
        decay_iter = num_epochs - warmup_epochs
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group.get('name') == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


# 需要的 import
import torch.nn as nn