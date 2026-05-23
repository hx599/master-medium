# -*- coding: utf-8 -*-
"""
分割模型定义 - 持续学习版本。

基于 HZSCM 的 HSI_Seg_Hard 改进：
1. 双分支分类头：已知类分支(固定) + 未知类分支(可扩展)
2. 跨模态注意力融合模块
3. 模态适配器

变量约定：
- `x`: 输入数据 [B, C, H, W]
- `known_logits`: 已知类分类头输出 [B, H, W, n_known_classes]
- `unknown_logits`: 未知类分类头输出 [B, H, W, n_unknown_classes]
- `all_logits`: 拼接后的完整输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

criterion = nn.CrossEntropyLoss(ignore_index=-1)
KL_loss = nn.KLDivLoss(reduction='batchmean')


def compute_using_CEloss(predict: torch.Tensor, gt: torch.Tensor):
    """计算像素级交叉熵损失。"""
    predict = rearrange(predict, 'b m n c -> (b m n) c')
    real_labels = rearrange(gt, 'b m n -> (b m n)')
    return criterion(predict, real_labels)


def compute_distill_loss(student_logits, teacher_logits, temperature=2.0, alpha=0.5):
    """
    计算知识蒸馏损失 (KL散度)。

    参数:
        student_logits: 学生模型输出 [B, H, W, C]
        teacher_logits: 教师模型输出 [B, H, W, C]
        temperature: 蒸馏温度
        alpha: 蒸馏损失权重
    """
    # 展平空间维度
    student_flat = rearrange(student_logits, 'b h w c -> (b h w) c')
    teacher_flat = rearrange(teacher_logits, 'b h w c -> (b h w) c')

    # 软目标蒸馏
    T = temperature
    student_soft = F.log_softmax(student_flat / T, dim=-1)
    teacher_soft = F.softmax(teacher_flat / T, dim=-1)

    distill_kl = KL_loss(student_soft, teacher_soft) * (T * T)

    return distill_kl


class CrossModalAttention(nn.Module):
    """
    跨模态注意力融合模块。

    用主模态的查询(Query)去关注辅助模态的键值对(Key-Value)，
    实现信息从辅助模态向主模态的选择性传递。
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 主模态的 Query
        self.q_proj = nn.Linear(dim, dim)
        # 辅助模态的 Key 和 Value
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # 输出投影 + dropout
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, primary_feat, aux_feat):
        """
        参数:
            primary_feat: 主模态特征 [B, C, H, W]
            aux_feat: 辅助模态特征 [B, C, H, W]

        返回:
            融合后的特征 [B, C, H, W]
        """
        B, C, H, W = primary_feat.shape

        # 变换维度: [B, C, H, W] -> [B, H*W, C]
        primary_flat = primary_feat.flatten(2).transpose(1, 2)
        aux_flat = aux_feat.flatten(2).transpose(1, 2)

        # 投影到 Q, K, V
        q = self.q_proj(primary_flat).reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(aux_flat).reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(aux_flat).reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 注意力加权求和
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)

        # 残差连接 + 维度变换: [B, H*W, C] -> [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class ModalityAdapter(nn.Module):
    """
    模态适配器模块。

    将非光学模态(SAR/LiDAR)特征投影到 CLIP 视觉编码器兼容的空间，
    解决模态与 CLIP 语义空间对齐困难的问题。
    """

    def __init__(self, in_channels, out_channels=768):
        super().__init__()
        # 轻量级投影头: 1x1卷积 + MLP
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GroupNorm(4, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        """
        参数:
            x: 输入特征 [B, C, H, W]

        返回:
            投影后的特征 [B, out_channels, H, W]
        """
        return self.project(x)


class U_Encoder_Layer(nn.Module):
    """U-Net 编码器层。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv_relu(x)
        x = self.downsample(x1)
        return x1, x


class U_Decoder_Layer(nn.Module):
    """U-Net 解码器层。"""

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), align_corners=True, mode='bilinear')
        x = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x)
        return x1


class HSI_UNet(nn.Module):
    """U-Net 主干网络，支持跨模态注意力融合。"""

    def __init__(self, in_channels, hidden_dim, out_dim, use_cross_attention=False, aux_channels=None, num_heads=4):
        super().__init__()
        self.use_cross_attention = use_cross_attention and aux_channels is not None

        # 初始投影
        self.sp_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),
        )

        # 辅助模态适配器(如果使用)
        if self.use_cross_attention and aux_channels is not None:
            self.aux_adapter = ModalityAdapter(aux_channels, hidden_dim)
            self.cross_attn = CrossModalAttention(hidden_dim, num_heads=num_heads)

        # 编码器
        self.encoder_layer0 = U_Encoder_Layer(hidden_dim, hidden_dim)
        self.encoder_layer1 = U_Encoder_Layer(hidden_dim, hidden_dim)
        self.encoder_layer2 = U_Encoder_Layer(hidden_dim, hidden_dim * 2)

        # 解码器
        self.decoder_layer2 = U_Decoder_Layer(hidden_dim * 2, hidden_dim * 4, hidden_dim)
        self.decoder_layer1 = U_Decoder_Layer(hidden_dim, hidden_dim * 2, hidden_dim)
        self.decoder_layer0 = U_Decoder_Layer(hidden_dim, hidden_dim * 2, hidden_dim)

        self.final_layer = nn.Conv2d(hidden_dim, out_dim, kernel_size=5, padding=2, bias=False)

    def forward(self, x, aux_x=None):
        """
        参数:
            x: 主模态输入 [B, C, H, W]
            aux_x: 辅助模态输入 [B, C_aux, H, W]，可选
        """
        spe_x = self.sp_net(x)

        # 跨模态注意力融合
        if self.use_cross_attention and aux_x is not None:
            aux_feat = self.aux_adapter(aux_x)
            spe_x = spe_x + self.cross_attn(spe_x, aux_feat)

        # 编码路径
        conv0, x = self.encoder_layer0(spe_x)
        conv1, x = self.encoder_layer1(x)
        conv2, x = self.encoder_layer2(x)

        # 解码路径
        x = self.decoder_layer2(x, conv2)
        x = self.decoder_layer1(x, conv1)
        x = self.decoder_layer0(x, conv0)

        # 输出: [B, D, H, W] -> [B, H, W, D]
        x = rearrange(self.final_layer(x), 'b d m n -> b m n d')
        return x


class DualBranchClassifier(nn.Module):
    """
    双分支分类头，支持持续学习。

    - 已知类分支：固定权重，防止遗忘
    - 未知类分支：可动态扩展
    """

    def __init__(self, in_dim, n_known_classes, n_total_classes):
        super().__init__()
        self.n_known_classes = n_known_classes
        self.n_total_classes = n_total_classes
        self.n_unknown_classes = n_total_classes - n_known_classes

        # 已知类分类头(可选择冻结)
        self.known_head = nn.Linear(in_dim, n_known_classes)

        # 未知类分类头(用于增量学习)
        self.unknown_head = nn.Linear(in_dim, self.n_unknown_classes)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.known_head.weight)
        nn.init.zeros_(self.known_head.bias)
        nn.init.xavier_uniform_(self.unknown_head.weight)
        nn.init.zeros_(self.unknown_head.bias)

    def freeze_known_head(self):
        """冻结已知类分类头，防止灾难性遗忘。"""
        for param in self.known_head.parameters():
            param.requires_grad = False

    def unfreeze_known_head(self):
        """解冻已知类分类头。"""
        for param in self.known_head.parameters():
            param.requires_grad = True

    def expand_unknown_head(self, new_num_classes):
        """
        扩展未知类分类头，支持增量学习。

        参数:
            new_num_classes: 新增类别数后的未知类总数
        """
        old_weight = self.unknown_head.weight.data
        old_bias = self.unknown_head.bias.data

        self.unknown_head = nn.Linear(self.unknown_head.in_features, new_num_classes).to(old_weight.device)

        # 复制旧权重到新头部的前面
        with torch.no_grad():
            self.unknown_head.weight[:self.n_unknown_classes] = old_weight
            self.unknown_head.bias[:self.n_unknown_classes] = old_bias

        self.n_unknown_classes = new_num_classes

    def forward(self, x):
        """
        参数:
            x: 特征 [B, H, W, D]

        返回:
            known_logits: 已知类 logits [B, H, W, n_known]
            unknown_logits: 未知类 logits [B, H, W, n_unknown]
        """
        B, H, W, D = x.shape
        x_flat = x.reshape(B * H * W, D)

        known_logits = self.known_head(x_flat).reshape(B, H, W, self.n_known_classes)
        unknown_logits = self.unknown_head(x_flat).reshape(B, H, W, self.n_unknown_classes)

        return known_logits, unknown_logits

    def get_all_logits(self, x):
        """获取拼接后的完整 logits。"""
        known_logits, unknown_logits = self.forward(x)
        return torch.cat([known_logits, unknown_logits], dim=-1)


class HSI_Seg_Incremental(nn.Module):
    """
    持续学习分割网络。

    支持:
    - 双分支分类头(防遗忘)
    - 跨模态注意力融合
    - 知识蒸馏
    - 增量类别扩展
    """

    def __init__(self, in_channels, hidden_dim, out_dim, n_known_classes, n_total_classes,
                 aux_channels=None, use_cross_attention=False, num_heads=4):
        super().__init__()
        self.n_known_classes = n_known_classes
        self.n_total_classes = n_total_classes

        # U-Net 主干
        self.backbone = HSI_UNet(
            in_channels, hidden_dim, out_dim,
            use_cross_attention=use_cross_attention,
            aux_channels=aux_channels,
            num_heads=num_heads
        )

        # 双分支分类头
        self.classifier = DualBranchClassifier(out_dim, n_known_classes, n_total_classes)

    def pre_forward(self, x, gt, aux_x=None):
        """
        第一阶段：仅使用已知类标签预训练。

        参数:
            x: 主模态输入
            gt: 真实标签
            aux_x: 辅助模态输入(可选)

        返回:
            输出logits和损失
        """
        x = self.backbone(x, aux_x)
        out = self.classifier.get_all_logits(x)

        # 标签转为0-based
        loss_l = compute_using_CEloss(out, gt - 1)

        return out, loss_l

    def forward(self, x, pseudo_label, gt, aux_x=None, distill_logits=None, distill_weight=0.5, temperature=2.0):
        """
        第二阶段：监督损失 + 伪标签损失 + 知识蒸馏损失。

        参数:
            x: 主模态输入
            pseudo_label: 伪标签
            gt: 真实标签
            aux_x: 辅助模态输入(可选)
            distill_logits: 教师模型logits，用于蒸馏
            distill_weight: 蒸馏损失权重
            temperature: 蒸馏温度

        返回:
            输出logits、监督损失、伪标签损失、蒸馏损失
        """
        x = self.backbone(x, aux_x)
        out = self.classifier.get_all_logits(x)

        # 监督损失
        loss_l = compute_using_CEloss(out, gt - 1)

        # 伪标签损失
        loss_u = compute_using_CEloss(out, pseudo_label)

        # 知识蒸馏损失
        loss_distill = 0.0
        if distill_logits is not None:
            loss_distill = compute_distill_loss(out, distill_logits, temperature=temperature)

        # 总损失
        loss = loss_l + args.beta * loss_u + distill_weight * loss_distill

        return out, loss_l, loss_u, loss_distill

    def test(self, x, aux_x=None):
        """
        测试时返回特征和logits。
        """
        x = self.backbone(x, aux_x)
        out = self.classifier.get_all_logits(x)
        return x, out

    def expand_classes(self, new_n_total_classes):
        """
        增量学习：扩展分类头以支持新类别。

        参数:
            new_n_total_classes: 新的总类别数
        """
        self.n_total_classes = new_n_total_classes
        self.classifier.expand_unknown_head(new_n_total_classes - self.n_known_classes)


# 全局变量用于forward中访问beta
args = None

def set_args(config):
    """设置全局配置。"""
    global args
    args = config