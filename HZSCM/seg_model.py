# -*- coding: utf-8 -*-
"""
分割模型定义。

整个模型很简单：
1. 先用一个轻量 U-Net 风格主干提取像素级特征
2. 再用线性分类头把每个像素特征映射成类别 logits
3. 在第二阶段训练时，同时使用真实标签和伪标签监督
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

criterion = nn.CrossEntropyLoss(ignore_index=-1)
KL_loss = nn.KLDivLoss(reduction='batchmean')


def compute_using_CEloss(predict: torch.Tensor, gt: torch.Tensor):
    """把二维像素网格拉平成一维后，计算像素级交叉熵损失。"""
    # predict: [B, H, W, C] -> [(B*H*W), C]
    predict = rearrange(predict, 'b m n c -> (b m n) c')
    # gt: [B, H, W] -> [(B*H*W)]
    real_labels = rearrange(gt, 'b m n -> (b m n)')
    pool_cross_entropy = criterion(predict, real_labels)
    return pool_cross_entropy



class U_Encoder_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U_Encoder_Layer, self).__init__()
        # 每个编码块都会先提取特征，再把空间尺寸减半。
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride = 2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),  # 加入Bn层提高网络泛化能力（防止过拟合），加收敛速度 out_channels//16
            nn.ReLU(inplace=True)
            )

    def forward(self, x):       
        # x1 用作跳跃连接，x 继续送入更深层的编码器。
        x1 = self.conv_relu(x)
        x = self.downsample(x1)        
        return x1, x


class U_Decoder_Layer(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(U_Decoder_Layer, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):       
        # x1：低分辨率解码特征
        # x2：对应尺度的编码器跳跃连接特征
        x1 = self.upsample(x1)
        # 这里再插值一次，保证和跳跃连接特征的空间尺寸完全对齐。
        x1 = F.interpolate(x1, size = (x2.size()[2], x2.size()[3]), align_corners=True, mode = 'bilinear')   
        x = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x)
        return x1


class HSI_UNet(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super().__init__()

        # 先用 1x1 卷积压缩输入通道数，减轻后续编码器的负担。
        self.sp_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, padding=0, bias = False),
            nn.GroupNorm(4, hidden_dim), 
            nn.ReLU(),
        )
        
        self.encoder_layer0 = U_Encoder_Layer(hidden_dim, hidden_dim)
        self.encoder_layer1 = U_Encoder_Layer(hidden_dim, hidden_dim)
        self.encoder_layer2 = U_Encoder_Layer(hidden_dim, hidden_dim*2)
        
        self.decoder_layer2 = U_Decoder_Layer(hidden_dim*2, hidden_dim*4, hidden_dim)
        self.decoder_layer1 = U_Decoder_Layer(hidden_dim, hidden_dim*2, hidden_dim)
        self.decoder_layer0 = U_Decoder_Layer(hidden_dim, hidden_dim*2, hidden_dim)
        
        self.final_layer = nn.Conv2d(hidden_dim, out_dim, kernel_size=5, padding=2, bias = False)


    def forward(self, x):
        # 进入网络前，patch 已经是 [B, C, H, W] 格式。
        spe_x = self.sp_net(x)

        # 编码路径：逐步扩大感受野。
        conv0, x = self.encoder_layer0(spe_x)
        conv1, x = self.encoder_layer1(x)
        conv2, x = self.encoder_layer2(x)
        
        # 解码路径：逐步恢复空间分辨率。
        x = self.decoder_layer2(x, conv2)
        x = self.decoder_layer1(x, conv1)
        x = self.decoder_layer0(x, conv0)
        
        # 输出转成 [B, H, W, D]，方便后面逐像素分类。
        x = rearrange(self.final_layer(x), 'b d m n -> b m n d')
        return x

class HSI_Seg_Hard(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim, nclass):
        super().__init__()
        self.backbone = HSI_UNet(in_channels, hidden_dim, out_dim)
        # 对每个像素特征都共享同一个线性分类器。
        self.projection_head = nn.Linear(out_dim, nclass)
        

    def pre_forward(self, x, gt):
        # 第一阶段：只使用真实标签做监督。
        x = self.backbone(x)
        out = self.projection_head(x)

        # 数据集标签是 1..C，但交叉熵要求类别编号从 0 开始。
        loss_l = compute_using_CEloss(out, gt - 1)

        return out, loss_l

    def forward(self, x, pseudo_label, gt):
        # 第二阶段：同时使用真实标签损失和伪标签损失。
        x = self.backbone(x)
        out = self.projection_head(x)


        loss_l = compute_using_CEloss(out, gt - 1)
        loss_u = compute_using_CEloss(out, pseudo_label)         
    
        return out, loss_l, loss_u

    def test(self, x):
        # 测试时同时返回像素特征和分类 logits，方便后续分析。
        x = self.backbone(x)
        out = self.projection_head(x)
        return x, out
