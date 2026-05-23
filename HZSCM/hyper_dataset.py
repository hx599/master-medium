# -*- coding: utf-8 -*-
"""训练和测试阶段使用的 PyTorch 数据集封装。"""

import numpy as np

import torch
from torch.utils.data.dataset import Dataset

class HyperData(Dataset):
    def __init__(self, dataset, transfor):
        # dataset = (图像 patch, 真实标签, 伪标签)
        self.data = dataset[0].astype(np.float32)
        self.transform = transfor
        self.labels = dataset[1]
        self.sp_labels = dataset[2]

    def __getitem__(self, index):
        # 每个样本本质上就是从整幅图中切出来的一个 patch。
        label = self.labels[index]
        sp_label = self.sp_labels[index]
        if self.transform == None:
            img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
            return img, label, sp_label
        else:
            # 当前项目没有实际走这条分支，
            # 但保留它可以兼容对比学习式的双视图增强。
            img_q, img_k = self.transform[0](self.data[index,:,:,:])  
            return [torch.from_numpy(np.asarray(img_q)), torch.from_numpy(np.asarray(img_k))], label, sp_label           

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels
    
class HyperDataPL(Dataset):
    def __init__(self, dataset, transfor):
        # dataset = (图像 patch, 真实标签, 伪标签, 置信度分数)
        self.data = dataset[0].astype(np.float32)
        self.transform = transfor
        self.labels = dataset[1]
        self.pl_labels = dataset[2]
        self.scores = dataset[3]

    def __getitem__(self, index):
        # 与 HyperData 类似，只是额外返回一个分数。
        label = self.labels[index]
        pl_label = self.pl_labels[index]
        score = self.scores[index]
        if self.transform == None:
            img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
            return img, label, pl_label, score
        else:
            img_q, img_k = self.transform[0](self.data[index,:,:,:])  
            return [torch.from_numpy(np.asarray(img_q)), torch.from_numpy(np.asarray(img_k))], label, pl_label, score        

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels   
    
