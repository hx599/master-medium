# -*- coding: utf-8 -*-
"""
实验参数入口。

这个文件只负责一件事：集中管理整个项目会用到的命令行参数。
这样后面复现实验、切换数据集、调整伪标签策略时，都只需要先看这里。
"""

import argparse


def _build_parser():
    """构建并返回整个项目的命令行参数解析器。"""
    parser = argparse.ArgumentParser()

    # 数据集名称，以及数据/中间产物所在目录。
    parser.add_argument('--dataset', type=str, default='MUUFL', help='Houston, HS-SAR-Berlin, MUUFL')
    parser.add_argument('--data_root', type=str, default='D:/Master_medium/dataset')
    parser.add_argument('--sp_root', type=str, default='D:/Master_medium/HZSCM/SP_mat')
    # single：训练时只输入主模态
    # early：训练时把多个模态在通道维直接拼接
    parser.add_argument('--fusion_mode', type=str, default='early', choices=['single', 'early'])
    # ratio：每类按比例采样；same_num：每类采样固定个数
    parser.add_argument('--samples_type', type=str, default='ratio')
    parser.add_argument('--curr_train_ratio', type=float, default=0.05)
    parser.add_argument('--curr_val_ratio', type=float, default=0)

    # 零样本设定：这里列出的类别会从监督标签中拿掉。
    parser.add_argument('--del_class', nargs='+', type=int, default=[1, 6],
                        help='unseen class indices (0-based), e.g. --del_class 1 6')

    # 如果整幅图太大，可以先切成多个 patch 再训练。
    parser.add_argument('--split_height', type=int, default=1)
    parser.add_argument('--split_width', type=int, default=1)
    # EDGE 表示 patch 周围额外保留多少像素边界，用来提供上下文。
    parser.add_argument('--EDGE', type=int, default=0)

    # 预训练阶段和正式训练阶段共用的优化参数。
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=600)
    parser.add_argument('--pre_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=2e-5)

    # 主干网络宽度，以及像素级特征输出维度。
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=128)

    # 伪标签生成相关参数。
    parser.add_argument('--margin', type=float, default=0.1, help='ease the boundary')
    parser.add_argument('--multi_clip', type=int, default=2, help='2: use two clip, 1:use rgb clip, 0: use rsicd clip')
    parser.add_argument('--use_sam', type=int, default=1, help='1: use sam, 0: only ERS')
    parser.add_argument('--pseudo_label_mode', type=str, default='generate', choices=['load', 'generate'],
                        help='load cached pseudo labels or generate and cache them')
    parser.add_argument('--use_multimodal_pseudo', type=int, default=0,
                        help='1: average pseudo-label probabilities from both modalities when available')
    parser.add_argument('--rsicd_clip_dir', type=str, default='D://Master_medium/Large model weights/pt_rsicd_clip_save_pretrained')
    parser.add_argument('--rgb_clip_dir', type=str, default='D://Master_medium//Large model weights/pt_clip-vit-large-patch14')

    # 基于超像素和局部聚合的伪标签修正参数。
    parser.add_argument('--tao', type=float, default=1.0, help='distance temprature in local aggregation')
    parser.add_argument('--n_neighbours', type=int, default=10, help='number of neighbours in clustering')
    parser.add_argument('--n_clusters', type=int, default=48, help='number of clusters in clustering')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold for noisy label detection in local aggregation')

    parser.add_argument('--logit_scale', type=float, default=1.0, help='temprature')

    # 正式训练阶段中，伪标签损失所占的权重。
    parser.add_argument('--beta', type=float, default=1.0, help='pesudo loss ratio')

    return parser


def load_args():
    """从命令行解析并返回实验参数。"""
    return _build_parser().parse_args()


def create_default_args(**overrides):
    """程序化创建参数对象，用于批量实验脚本。

    用法: args = create_default_args(dataset='Houston', del_class=[1, 6], epoch=300)
    """
    args = _build_parser().parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f'Unknown argument: {key}')
        setattr(args, key, value)
    return args
