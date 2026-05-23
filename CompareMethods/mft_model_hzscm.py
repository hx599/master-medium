import copy

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import Dropout, LayerNorm, Linear


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, p=64, g=64):
        super().__init__()
        self.gwc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=g,
            padding=kernel_size // 3,
            stride=stride,
        )
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, stride=stride)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        bsz, tokens, dim = x.shape
        q = self.wq(x[:, 0:1, ...].reshape(bsz, 1, self.num_heads, dim // self.num_heads)).permute(0, 2, 1, 3)
        k = self.wk(x.reshape(bsz, tokens, self.num_heads, dim // self.num_heads)).permute(0, 2, 1, 3)
        v = self.wv(x.reshape(bsz, tokens, self.num_heads, dim // self.num_heads)).permute(0, 2, 1, 3)
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2)
        x = x.reshape(bsz, 1, dim * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim)
        self.attn = MCrossAttention(dim=dim)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(2):
            self.layer.append(copy.deepcopy(Block(dim)))

    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)
        encoded = self.encoder_norm(x)
        return encoded[:, 0]


class MFT(nn.Module):
    def __init__(self, fm, num_hsi_bands, num_aux_bands, num_classes, patch_size=11):
        super().__init__()
        self.patch_size = patch_size
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            HetConv(
                8 * (num_hsi_bands - 8),
                fm * 4,
                p=1,
                g=(fm * 4) // 4 if (8 * (num_hsi_bands - 8)) % fm == 0 else (fm * 4) // 8,
            ),
            nn.BatchNorm2d(fm * 4),
            nn.ReLU(),
        )
        self.lidar_conv = nn.Sequential(
            nn.Conv2d(num_aux_bands, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.ca = TransformerEncoder(fm * 4)
        self.out3 = nn.Linear(fm * 4, num_classes)
        self.position_embeddings = nn.Parameter(torch.randn(1, 5, fm * 4))
        self.dropout = nn.Dropout(0.1)
        nn.init.xavier_uniform_(self.out3.weight)
        nn.init.normal_(self.out3.bias, std=1e-6)
        self.token_wA = nn.Parameter(torch.empty(1, 4, 64), requires_grad=True)
        nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64), requires_grad=True)
        nn.init.xavier_normal_(self.token_wV)
        self.token_wA_L = nn.Parameter(torch.empty(1, 1, 64), requires_grad=True)
        nn.init.xavier_normal_(self.token_wA_L)
        self.token_wV_L = nn.Parameter(torch.empty(1, 64, 64), requires_grad=True)
        nn.init.xavier_normal_(self.token_wV_L)

    def forward(self, x_hsi, x_aux):
        x_hsi = x_hsi.reshape(x_hsi.shape[0], -1, self.patch_size, self.patch_size)
        x_hsi = x_hsi.unsqueeze(1)
        x_aux = x_aux.reshape(x_aux.shape[0], -1, self.patch_size, self.patch_size)

        x_hsi = self.conv5(x_hsi)
        x_hsi = x_hsi.reshape(x_hsi.shape[0], -1, self.patch_size, self.patch_size)
        x_hsi = self.conv6(x_hsi)

        x_aux = self.lidar_conv(x_aux)
        x_aux = x_aux.reshape(x_aux.shape[0], -1, self.patch_size ** 2)
        x_aux = x_aux.transpose(-1, -2)
        wa_l = self.token_wA_L.expand(x_hsi.shape[0], -1, -1)
        wa_l = rearrange(wa_l, "b h w -> b w h")
        a_l = torch.einsum("bij,bjk->bik", x_aux, wa_l)
        a_l = rearrange(a_l, "b h w -> b w h")
        a_l = a_l.softmax(dim=-1)
        wv_l = self.token_wV_L.expand(x_aux.shape[0], -1, -1)
        vv_l = torch.einsum("bij,bjk->bik", x_aux, wv_l)
        x_aux = torch.einsum("bij,bjk->bik", a_l, vv_l)

        x_hsi = x_hsi.flatten(2)
        x_hsi = x_hsi.transpose(-1, -2)
        wa = self.token_wA.expand(x_hsi.shape[0], -1, -1)
        wa = rearrange(wa, "b h w -> b w h")
        a = torch.einsum("bij,bjk->bik", x_hsi, wa)
        a = rearrange(a, "b h w -> b w h")
        a = a.softmax(dim=-1)
        wv = self.token_wV.expand(x_hsi.shape[0], -1, -1)
        vv = torch.einsum("bij,bjk->bik", x_hsi, wv)
        tokens = torch.einsum("bij,bjk->bik", a, vv)

        x = torch.cat((x_aux, tokens), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        x = self.ca(x)
        x = x.reshape(x.shape[0], -1)
        return self.out3(x)
