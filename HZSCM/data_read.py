# -*- coding: utf-8 -*-
"""
统一的数据读取入口。

项目其它地方只需要导入这个文件里的 `read_data()`。
它会根据 `fusion_mode` 自动分发到：
1. 单模态训练版本
2. 早期融合的多模态训练版本
"""

from data_read_multimodal import read_data as read_data_multimodal
from data_read_singlemodal import read_data as read_data_singlemodal


def read_data(args, curr_seed):
    """根据 fusion_mode 选择对应的数据处理实现。"""
    fusion_mode = getattr(args, 'fusion_mode', 'early')
    if fusion_mode == 'early':
        return read_data_multimodal(args, curr_seed)
    return read_data_singlemodal(args, curr_seed)
