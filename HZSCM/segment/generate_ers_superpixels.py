# -*- coding: utf-8 -*-
"""
生成 ERS 风格的超像素标签，并保存成项目当前使用的 `.mat` 文件。

脚本优先调用你本机已经安装的 ERS Python 后端；
如果没有找到严格 ERS 实现，可以按参数设置退化为 SLIC，
先把整条伪标签生成流程跑通。

对双模态数据集，默认会分别输出：
- `<dataset>_primary_SPGT_<N>.mat`
- `<dataset>_secondary_SPGT_<N>.mat`
"""

import argparse
from pathlib import Path
import sys

import numpy as np

try:
    from segment.segmentation_dataset_utils import (
        available_modalities,
        dataset_rgb_image,
        default_ers_output_path,
        resolve_dataset_name,
        save_superpixel_labels,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from segmentation_dataset_utils import (
        available_modalities,
        dataset_rgb_image,
        default_ers_output_path,
        resolve_dataset_name,
        save_superpixel_labels,
    )

try:
    from skimage.segmentation import slic
except ImportError:
    slic = None


def build_parser():
    """定义命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HS-SAR-Berlin', help='Houston, Berlin, HS-SAR-Berlin, MUUFL')
    parser.add_argument('--data_root', type=str, default='D:/Master_medium/dataset')
    parser.add_argument('--sp_root', type=str, default='./SP_mat')
    parser.add_argument('--modality', type=str, default='both', choices=['primary', 'secondary', 'both'])
    parser.add_argument('--n_segments', type=int, default=2000)
    parser.add_argument('--lambda_value', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=5.0)
    parser.add_argument('--connectivity', type=int, default=8, choices=[4, 8])
    parser.add_argument('--allow_fallback', type=int, default=1,
                        help='1: allow SLIC fallback when an exact ERS backend is unavailable')
    parser.add_argument('--output', type=str, default=None)
    return parser


def _try_import_ers_backend():
    """尝试导入本机可能存在的 ERS Python 包。"""
    module_names = ['erspy', 'pyers', 'ers']
    for module_name in module_names:
        try:
            module = __import__(module_name)
            return module_name, module
        except ImportError:
            continue
    return None, None


def _call_ers_backend(backend_module, image, n_segments, lambda_value, sigma, connectivity):
    """
    兼容不同 ERS 包可能存在的函数名和参数签名。

    不同实现的 API 往往不统一，所以这里按常见名字依次尝试调用。
    """
    call_specs = [
        ('run', (image, n_segments, lambda_value, sigma, connectivity)),
        ('ers', (image, n_segments, lambda_value, sigma, connectivity)),
        ('segment', (image, n_segments, lambda_value, sigma, connectivity)),
        ('computeSegmentation', (image, n_segments, lambda_value, sigma, connectivity)),
        ('run', (image, n_segments)),
        ('ers', (image, n_segments)),
        ('segment', (image, n_segments)),
        ('computeSegmentation', (image, n_segments)),
    ]
    for attr_name, call_args in call_specs:
        func = getattr(backend_module, attr_name, None)
        if func is None:
            continue
        try:
            labels = func(*call_args)
            return np.asarray(labels)
        except TypeError:
            continue
    raise RuntimeError(
        'Found an ERS backend module, but none of the expected call signatures worked. '
        'Please adapt _call_ers_backend() to your local ERS package.'
    )


def _generate_slic_fallback(image, n_segments):
    """当本机没有 ERS 后端时，使用 SLIC 作为临时替代方案。"""
    if slic is None:
        raise ImportError('scikit-image is required for the fallback path')
    return slic(
        image,
        n_segments=n_segments,
        compactness=10.0,
        sigma=1.0,
        start_label=0,
        channel_axis=-1,
    )


def generate_superpixels(args):
    """读取图像并生成超像素标签图。"""
    dataset_name = resolve_dataset_name(args.dataset)
    modalities = available_modalities(dataset_name, args.data_root)
    if args.modality != 'both':
        if args.modality not in modalities:
            raise ValueError(f'Dataset {dataset_name} does not provide modality {args.modality}')
        modalities = [args.modality]

    if args.output is not None and len(modalities) > 1:
        raise ValueError('When generating multiple modalities, please omit --output and use the default file names.')

    backend_name, backend_module = _try_import_ers_backend()
    print(f'dataset={dataset_name}')
    for modality in modalities:
        image = dataset_rgb_image(dataset_name, modality, args.data_root)
        image = np.ascontiguousarray(image)

        if backend_module is not None:
            # 优先使用真正的 ERS 后端。
            labels = _call_ers_backend(
                backend_module,
                image,
                args.n_segments,
                args.lambda_value,
                args.sigma,
                args.connectivity,
            )
            used_backend = f'ERS:{backend_name}'
        else:
            if not bool(args.allow_fallback):
                raise ImportError(
                    'No ERS backend was found. Install a local ERS Python backend such as erspy, '
                    'or rerun with --allow_fallback 1 to use SLIC as a temporary substitute.'
                )
            # 本机没有 ERS 后端时，用 SLIC 临时替代以保证流程可运行。
            labels = _generate_slic_fallback(image, args.n_segments)
            used_backend = 'fallback:SLIC'

        labels = np.asarray(labels, dtype=np.int32)
        if labels.ndim != 2:
            raise ValueError(f'Expected a 2D label map, but got shape {labels.shape}')

        # 保存成主项目默认读取的 `.mat` 文件格式。
        output_path = Path(args.output) if args.output else default_ers_output_path(
            dataset_name, args.sp_root, args.n_segments, modality=modality
        )
        save_superpixel_labels(labels, output_path)

        print(f'modality={modality}')
        print(f'backend={used_backend}')
        print(f'segments={len(np.unique(labels))}')
        print(f'output={output_path}')


def main():
    """脚本入口。"""
    parser = build_parser()
    args = parser.parse_args()
    generate_superpixels(args)


if __name__ == '__main__':
    main()
