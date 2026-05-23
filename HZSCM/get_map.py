# -*- coding: utf-8 -*-
"""把预测标签图渲染成彩色分类图。"""

import numpy as np
import sklearn

import matplotlib.pyplot as plt

indianpines_colors = np.array([[255,255, 255],
                                [255, 0, 0], [0,  255,  0], [0, 0,    255], [255,   255, 0],
                                [0,   255, 255], [255,  0,  255], [192,   192, 192], [128,  128,   128],
                                [128, 0,  0], [128, 128, 0], [0, 128, 0], [128,   0, 128],
                                [0, 128,  128], [0, 0,  128], [255, 165, 0], [255, 215,   0]])
indianpines_colors = sklearn.preprocessing.minmax_scale(indianpines_colors, feature_range=(0, 1))

def classification_map(img, dpi, save_path):
    """把彩色图像无坐标轴、无边框地保存到磁盘。"""
    fig = plt.figure(frameon=False)
    fig.set_size_inches(img.shape[1] * 2.0 / dpi, img.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(img)
    fig.savefig(save_path, dpi=dpi)
    
    return 0 


def generate(pred, path):  
    """把二维类别图映射成伪彩色专题图并保存。"""
    number_of_rows = np.size(pred,0)
    number_of_columns = np.size(pred,1)
     
    predicted_thematic_map = np.zeros(shape=(number_of_rows, number_of_columns, 3))
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            # 这里的预测标签是从 0 开始的，因此要加 1，
            # 以跳过调色板里索引 0 的白色背景。
            predicted_thematic_map[i, j, :] = indianpines_colors[pred[i,j]+1]  
    classification_map(predicted_thematic_map, 600,
                        path + '.png')    
    return predicted_thematic_map


