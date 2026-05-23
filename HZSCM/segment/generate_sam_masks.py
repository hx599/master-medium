# -*- coding: utf-8 -*-
"""
生成 SAM 自动分割结果，并保存成项目当前使用的 `.pkl` 文件。

输出文件可以直接被主项目读取，用于：
1. 和 ERS 超像素做融合；
2. 辅助后续的 CLIP 伪标签生成。

对双模态数据集，默认会分别输出：
- `<dataset>_primary_list.pkl`
- `<dataset>_secondary_list.pkl`
"""

import argparse
import pickle
from pathlib import Path

import sys
import numpy as np
import torch

try:
    from segment.segmentation_dataset_utils import (
        available_modalities,
        dataset_rgb_image,
        default_sam_output_path,
        resolve_dataset_name,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from segmentation_dataset_utils import (
        available_modalities,
        dataset_rgb_image,
        default_sam_output_path,
        resolve_dataset_name,
    )

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:
    SamAutomaticMaskGenerator = None
    sam_model_registry = None


def build_parser():
    """定义命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HS-SAR-Berlin', help='Houston, HS-SAR-Berlin, MUUFL')
    parser.add_argument('--data_root', type=str, default='D:/Master_medium/dataset')
    parser.add_argument('--sp_root', type=str, default='./SP_mat')
    parser.add_argument('--modality', type=str, default='both', choices=['primary', 'secondary', 'both'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='D:/Master_medium/HZSCM/segment/sam_vit_h_4b8939.pth', help='Path to the SAM checkpoint .pth')
    parser.add_argument('--model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--points_per_side', type=int, default=32)
    parser.add_argument('--pred_iou_thresh', type=float, default=0.88)
    parser.add_argument('--stability_score_thresh', type=float, default=0.95)
    parser.add_argument('--crop_n_layers', type=int, default=1)
    parser.add_argument('--crop_n_points_downscale_factor', type=int, default=2)
    parser.add_argument('--min_mask_region_area', type=int, default=100)
    return parser


def generate_sam_masks(args):
    """读取图像、运行 SAM，并把 mask 列表写入 pickle 文件。"""
    if SamAutomaticMaskGenerator is None or sam_model_registry is None:
        raise ImportError(
            'segment_anything is required. Install the official package from '
            'https://github.com/facebookresearch/segment-anything'
        )

    dataset_name = resolve_dataset_name(args.dataset)
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        # 当前机器没有可用 GPU 时，自动退回 CPU。
        device = 'cpu'

    # `both` 模式下会自动为该数据集的每个可用模态都生成一份 SAM 结果。
    modalities = available_modalities(dataset_name, args.data_root)
    if args.modality != 'both':
        if args.modality not in modalities:
            raise ValueError(f'Dataset {dataset_name} does not provide modality {args.modality}')
        modalities = [args.modality]

    if args.output is not None and len(modalities) > 1:
        raise ValueError('When generating multiple modalities, please omit --output and use the default file names.')

    # 加载指定版本的 SAM 权重。
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    # 通过自动 mask 生成器一次性产生整幅图的候选区域。
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
    )

    print(f'dataset={dataset_name}')
    print(f'device={device}')
    for modality in modalities:
        # 根据主模态/辅助模态配置，生成一张供 SAM 使用的 RGB 图。
        image = dataset_rgb_image(dataset_name, modality, args.data_root)
        image = np.ascontiguousarray(image)
        masks = mask_generator.generate(image)

        # 若未手动指定输出路径，则使用项目内部约定的默认命名。
        output_path = Path(args.output) if args.output else default_sam_output_path(
            dataset_name, args.sp_root, modality=modality
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as file:
            pickle.dump(masks, file)

        print(f'modality={modality}')
        print(f'masks={len(masks)}')
        print(f'output={output_path}')


def main():
    """脚本入口。"""
    parser = build_parser()
    args = parser.parse_args()
    generate_sam_masks(args)


if __name__ == '__main__':
    main()
