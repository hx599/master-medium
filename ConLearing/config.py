# -*- coding: utf-8 -*-
"""
实验参数入口 - 持续学习版本。

在 HZSCM 基础上新增持续学习相关参数：
- 增量类别配置
- 知识蒸馏参数
- 模态适配器参数
- 跨模态注意力参数
"""

import argparse


def _build_parser():
    """构建并返回实验参数解析器。"""
    parser = argparse.ArgumentParser()

    # ==================== 数据集基础参数 ====================
    parser.add_argument('--dataset', type=str, default='MUUFL', help='Houston, HS-SAR-Berlin, MUUFL')
    parser.add_argument('--data_root', type=str, default='D:/Master_medium/dataset')
    parser.add_argument('--sp_root', type=str, default='D:/Master_medium/HZSCM/SP_mat')

    # 融合模式
    parser.add_argument('--fusion_mode', type=str, default='early', choices=['single', 'early', 'attention'])
    parser.add_argument('--samples_type', type=str, default='ratio')
    parser.add_argument('--curr_train_ratio', type=float, default=0.05)
    parser.add_argument('--curr_val_ratio', type=float, default=0)

    # 零样本设定
    parser.add_argument('--del_class', nargs='+', type=int, default=[1, 6],
                        help='unseen class indices (0-based), e.g. --del_class 1 6')

    # 图像分块参数
    parser.add_argument('--split_height', type=int, default=1)
    parser.add_argument('--split_width', type=int, default=1)
    parser.add_argument('--EDGE', type=int, default=0)

    # ==================== 训练参数 ====================
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=600)
    parser.add_argument('--pre_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=2e-5)

    # 模型结构参数
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=128)

    # ==================== 伪标签参数 ====================
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--multi_clip', type=int, default=2, help='2: use two clip, 1:use rgb clip, 0: use rsicd clip')
    parser.add_argument('--use_sam', type=int, default=1, help='1: use sam, 0: only ERS')
    parser.add_argument('--pseudo_label_mode', type=str, default='generate', choices=['load', 'generate'])
    parser.add_argument('--use_multimodal_pseudo', type=int, default=1)
    parser.add_argument('--rsicd_clip_dir', type=str, default='D://Master_medium/Large model weights/pt_rsicd_clip_save_pretrained')
    parser.add_argument('--rgb_clip_dir', type=str, default='D://Master_medium//Large model weights/pt_clip-vit-large-patch14')

    # 伪标签修正参数
    parser.add_argument('--tao', type=float, default=1.0)
    parser.add_argument('--n_neighbours', type=int, default=10)
    parser.add_argument('--n_clusters', type=int, default=48)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--logit_scale', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0, help='pseudo loss ratio')

    # ==================== 持续学习参数 ====================
    # 增量类别配置 (新增)
    parser.add_argument('--incremental_classes', nargs='+', type=int, default=[],
                        help='新增的类别列表，用于持续学习场景')
    parser.add_argument('--num_increments', type=int, default=1,
                        help='增量学习的阶段数')
    parser.add_argument('--classes_per_increment', type=int, default=2,
                        help='每个增量阶段引入的类别数')

    # 知识蒸馏参数 (新增)
    parser.add_argument('--distill_weight', type=float, default=0.5,
                        help='知识蒸馏损失权重')
    parser.add_argument('--distill_temperature', type=float, default=2.0,
                        help='知识蒸馏温度')
    parser.add_argument('--freeze_known_head', type=True, type=bool, default=True,
                        help='增量训练时冻结已知类分类头')

    # 模态适配器参数 (新增)
    parser.add_argument('--use_modality_adapter', type=int, default=1,
                        help='是否使用模态适配器')
    parser.add_argument('--adapter_dim', type=int, default=128,
                        help='模态适配器中间层维度')

    # 跨模态注意力参数 (新增)
    parser.add_argument('--use_cross_attention', type=int, default=1,
                        help='是否使用跨模态注意力融合')
    parser.add_argument('--num_attention_heads', type=int, default=4,
                        help='跨模态注意力头数')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='注意力 dropout')

    # 保存路径
    parser.add_argument('--checkpoint_dir', type=str, default='./ConLearing/checkpoints')
    parser.add_argument('--save_interval', type=int, default=50)

    return parser


def load_args():
    """从命令行解析并返回实验参数。"""
    return _build_parser().parse_args()


def create_default_args(**overrides):
    """程序化创建参数对象，用于批量实验脚本。"""
    args = _build_parser().parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f'Unknown argument: {key}')
        setattr(args, key, value)
    return args