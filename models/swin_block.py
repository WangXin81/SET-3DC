import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

##single_Swin_block

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, H//window_size, W//window_size, window_size, window_size, C]
    x = x.reshape(-1, window_size, window_size, C)  # [B*num_window, window_size, window_size, C]

    return x


def window_reverse(windows, window_size, H, W):
    # windows:[B*num_window, window_size, window_size, C]
    B = int(windows.shape[0] // (H / window_size * W / window_size))
    # x: [B, H//window_size, W//window_size, window_size, window_size, C]
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, H//window_size, window_size, W//window_size, window_size, C]

    x = x.reshape(B, H, W, -1)
    return x


def generate_mask(input_res, window_size, shift_size):
    H, W, = input_res
    Hp = int(np.ceil(H / window_size)) * window_size
    Wp = int(np.ceil(W / window_size)) * window_size

    image_mask = torch.zeros((1, Hp, Wp, 1))
    h_slice = (slice(0, -window_size),
               slice(-window_size, -shift_size),
               slice(-shift_size, None)
               )

    w_slice = (slice(0, -window_size),
               slice(-window_size, -shift_size),
               slice(-shift_size, None)
               )

    cnt = 0
    for h in h_slice:
        for w in w_slice:
            image_mask[:, h, w, :] = cnt
            cnt += 1
    mask_window = window_partition(image_mask, window_size)
    mask_window = mask_window.reshape(-1, window_size * window_size)

    attn_mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class Patch_Embeding(nn.Module):
    def __init__(self, chan=3, dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.patch = nn.Conv2d(chan, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if self.patch_size != 0:
            x = self.patch(x)  # [B, C, H, W] , C = dim
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, C]
        x = self.norm(x)
        return x


class Patch_Merging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.resolution = input_res
        self.dim = dim

        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x):
        # x: [B, num_patches, C]
        # H, W = self.resolution
        B, C, H, W = x.shape

        x = x.reshape(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat((x0, x1, x2, x3), -1)

        x = x.reshape(B, -1, 4 * C)
        x = self.reduction(x)
        x = self.norm(x)

        return x, H//2, W//2


# Swin_block


class window_attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop = 0.0, dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        self.inner_dim = dim
        prehead_dim = dim // self.num_heads
        self.scale = prehead_dim ** -0.5

        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: [B*num_window, num_patches, embed_dim]
        B, num_patches, total_dim = x.shape

        qkv = self.qkv(x)  # [B*num_window,, num_patches, 3*embed_dim]

        qkv = qkv.reshape(B, num_patches, 3, self.num_heads,
                          self.inner_dim // self.num_heads)  # [B*num_window,, num_patches, 3, num_heads, prehead_dim]

        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*num_window,, num_heads, num_patches, prehead_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B*num_window,, num_heads, num_patches, prehead_dim]

        atten = (q @ k.transpose(-2, -1)) * self.scale  # [B*num_window,, num_heads, num_patches, num_patches]
        if mask is None:
            atten = atten.softmax(dim=-1)
        else:
            # mask: [num_window, num_patches, num_patches]
            # atten: [B*num_window, num_head, num_patches, num_patches]
            atten = atten.reshape(B // mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1])
            # reshape_atten [B, num_window, num_head, num_patches, num_patches]
            # mask [1, num_window, 1, num_patches, num_patches]
            atten = atten + mask.unsqueeze(1).unsqueeze(0).cuda()  # atten = atten + mask.unsqueeze(1).unsqueeze(0)
            atten = atten.reshape(-1, self.num_heads, mask.shape[1],
                                  mask.shape[1])  # [B*num_window, num_head, num_patches, num_patches]
            atten = atten.softmax(dim=-1)

        atten = self.attn_drop(atten)
        atten = atten @ v  ## [B, num_heads, num_patches, prehead_dim]
        atten = atten.transpose(1, 2)  # [B, num_patches+1, num_heads, prehead_dim]
        atten = atten.reshape(B, num_patches, self.inner_dim)  # [B, num_patches+1, embed_dim]


        out = self.proj(atten)

        return out


class cross_window_attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        inner_dim = dim
        self.inner_dim = inner_dim
        prehead_dim = dim // self.num_heads
        self.scale = prehead_dim ** -0.5

        self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2, mask=None):
        # x: [B*num_window, num_patches, embed_dim]
        B, num_patches, total_dim, h = x1.shape[0], x1.shape[1], x1.shape[2], self.num_heads

        q = self.q(x2)  # [B*num_window,, num_patches, 3*embed_dim]
        k = self.q(x1)
        v = self.q(x1)

        q, k, v = map(lambda t: rearrange(t, 'b p (h d) -> b h p d', h=h, p=num_patches),
                      [q, k, v])  # [B*num_window,, num_patches, 3, num_heads, prehead_dim]

        # qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*num_window,, num_heads, num_patches, prehead_dim]
        # q, k, v = qkv[0], qkv[1], qkv[2]  # [B*num_window,, num_heads, num_patches, prehead_dim]

        atten = (q @ k.transpose(-2, -1)) * self.scale  # [B*num_window,, num_heads, num_patches, num_patches]
        if mask is None:
            atten = atten.softmax(dim=-1)
        else:
            # mask: [num_window, num_patches, num_patches]
            # atten: [B*num_window, num_head, num_patches, num_patches]
            atten = atten.reshape(B // mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1])
            # reshape_atten [B, num_window, num_head, num_patches, num_patches]
            # mask [1, num_window, 1, num_patches, num_patches]
            atten = atten + mask.unsqueeze(1).unsqueeze(0).cuda()  # atten = atten + mask.unsqueeze(1).unsqueeze(0)
            atten = atten.reshape(-1, self.num_heads, mask.shape[1],
                                  mask.shape[1])  # [B*num_window, num_head, num_patches, num_patches]
            atten = atten.softmax(dim=-1)

        atten = atten @ v  ## [B, num_heads, num_patches, prehead_dim]
        atten = atten.transpose(1, 2)  # [B, num_patches+1, num_heads, prehead_dim]
        atten = atten.reshape(B, num_patches, self.inner_dim)  # [B, num_patches+1, embed_dim]

        out = self.proj(atten)

        return out


class MLP(nn.Module):
    def __init__(self, in_dim, dropout, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim * mlp_ratio)
        self.dwconv = DWConv(in_dim * mlp_ratio)
        self.actlayer = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_ratio * in_dim, in_dim)

    def forward(self, x, H, W):
        x = self.fc1(x)  # [B, num_patches+1, hidden_dim]
        x = self.dwconv(x, H, W)
        x = self.actlayer(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_patches+1, out_dim]
        x = self.dropout(x)

        # x = self.actlayer(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# swin_encode & Patch_Merging
class Swin_Block(nn.Module):
    def __init__(self, dim,  num_heads, window_size, mlp_ratio,  dropout=0.0, qkv_bias=False, shift_size=0, attn_drop = 0., drop_path = 0.):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.shift_size = shift_size
        self.atten_norm = nn.LayerNorm(dim)
        self.atten = window_attention(dim, num_heads, qkv_bias, attn_drop=attn_drop, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout=dropout, mlp_ratio=mlp_ratio)
        # self.to_patch_embedding = nn.Conv2d(in_channels=dim,
        #                                  out_channels=dim,
        #                                  kernel_size=patch_size,
        #                                  stride=patch_size)


        # self.patch_merging = Patch_Merging(input_res, dim)

    def forward(self, x, H, W):
        # x:[B, num_patches, embed_dim]
        B, N, C = x.shape
        assert N == H * W

        h = x
        x = self.atten_norm(x)
        x = x.reshape(B, H, W, C)

        if self.shift_size > 0:
            shift_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            atten_mask = generate_mask(input_res=(H, W), window_size=self.window_size,
                                       shift_size=self.shift_size)
        else:
            shift_x = x
            atten_mask = None

        x_window = window_partition(shift_x, self.window_size)  # [B*num_patches, window_size, window_size, C]
        x_window = x_window.reshape(-1, self.window_size * self.window_size, C)
        atten_window = self.atten(x_window, mask=atten_mask)  # [B*num_patches, window_size*window_size, C]
        atten_window = atten_window.reshape(-1, self.window_size, self.window_size, C)
        x = window_reverse(atten_window, self.window_size, H, W)  # [B, H, W, C]
        x = x.reshape(B, -1, C)
        x = h + self.drop_path(x)

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x, H, W)
        x = h + self.drop_path(x)
        
        # x = x.reshape(B, -1, H, W)
        # x = self.to_patch_embedding(x)
        # x = x.reshape(B, -1, C)

        return x


class cross_Swin_Block(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio, qkv_bias=False, shift_size=0, dropout=0.0, drop_path = 0.):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.atten_norm = nn.LayerNorm(dim)
        self.atten = cross_window_attention(dim, num_heads, qkv_bias, dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout=dropout, mlp_ratio=mlp_ratio)

        # self.patch_merging = Patch_Merging(input_res, dim)

    def forward(self, x1, x2, H, W):
        # x:[B, num_patches, embed_dim]
        B, N, C = x1.shape
        assert N == H * W

        h = x1
        x1 = self.atten_norm(x1)
        x2 = self.atten_norm(x2)
        x1 = x1.reshape(B, H, W, C)
        x2 = x2.reshape(B, H, W, C)

        if self.shift_size > 0:
            shift_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shift_x2 = torch.roll(x2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            atten_mask = generate_mask(input_res=(H, W), window_size=self.window_size,
                                       shift_size=self.shift_size)
        else:
            shift_x1 = x1
            shift_x2 = x2
            atten_mask = None

        x_window1 = window_partition(shift_x1, self.window_size)
        x_window2 = window_partition(shift_x2, self.window_size)
        # [B*num_patches, window_size, window_size, C]
        x_window1 = x_window1.reshape(-1, self.window_size * self.window_size, C)
        x_window2 = x_window2.reshape(-1, self.window_size * self.window_size, C)
        atten_window = self.atten(x_window1, x_window2, mask=atten_mask)  # [B*num_patches, window_size*window_size, C]
        atten_window = atten_window.reshape(-1, self.window_size, self.window_size, C)
        x = window_reverse(atten_window, self.window_size, H, W)  # [B, H, W, C]
        x = x.reshape(B, -1, C)
        x = h + self.drop_path(x)

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x, H, W)
        x = h + self.drop_path(x)

        return x


class Swin_Model(nn.Module):
    def __init__(self,
                 chan,
                 dim,
                 dim_head,
                 depth,
                 patch_size,
                 num_heads,
                 input_res,
                 window_size,
                 qkv_bias=None,
                 dropout=0.0
                 ):
        super().__init__()

        self.patch_size = patch_size
        self.depth = depth
        self.patch_embed = Patch_Embeding(chan, dim, patch_size=patch_size)
        self.W_MSA_block = Swin_Block(dim, dim_head, num_heads, input_res, window_size, dropout, qkv_bias, shift_size=0)
        self.SW_MSA_block = Swin_Block(dim, dim_head, num_heads, input_res, window_size, dropout, qkv_bias,
                                       shift_size=window_size // 2)
        self.patch_merging = Patch_Merging(dim)

    def forward(self, x):
        x = self.patch_embed(x)
        for i in range(self.depth):
            x = self.W_MSA_block(x)
            x = self.SW_MSA_block(x)
        out = self.patch_merging(x)

        h, w = int(np.sqrt(out.size(1))), int(np.sqrt(out.size(1)))
        x = out.permute(0, 2, 1)
        x = x.contiguous().view(out.size(0), out.size(2), h, w)

        return x


class cross_Swin_Model(nn.Module):
    def __init__(self,
                 chan,
                 dim,
                 dim_head,
                 patch_size,
                 num_heads,
                 input_res,
                 window_size,
                 qkv_bias=None
                 ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_embed = Patch_Embeding(chan, dim, patch_size=patch_size)
        self.W_MSA_block = cross_Swin_Block(dim, dim_head, num_heads, input_res, window_size, qkv_bias, shift_size=0)
        self.SW_MSA_block = cross_Swin_Block(dim, dim_head, num_heads, input_res, window_size, qkv_bias,
                                             shift_size=window_size // 2)
        # self.patch_merging = Patch_Merging(input_res, dim)

    def forward(self, x1, x2):
        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        x11 = self.W_MSA_block(x1, x2)
        x22 = self.W_MSA_block(x2, x1)
        x111 = self.SW_MSA_block(x11, x22)
        x222 = self.SW_MSA_block(x22, x11)
        # out = self.patch_merging(x)

        h, w = int(np.sqrt(x1.size(1))), int(np.sqrt(x1.size(1)))

        x111 = x111.permute(0, 2, 1)
        x222 = x222.permute(0, 2, 1)
        x_1 = x111.contiguous().view(x111.size(0), x111.size(1), h, w)
        x_2 = x222.contiguous().view(x222.size(0), x222.size(1), h, w)
        x = x_1 - x_2

        return x


class swin_branch(nn.Module):
    def __init__(self, output_nc, dim_head, depth, dropout):
        super().__init__()
        self.swin_1 = Swin_Model(3, 32, dim_head, depth, 2, 8, (128, 128), 4, dropout=dropout)
        self.swin_2 = Swin_Model(64, 64, dim_head, depth, 1, 8, (64, 64), 4, dropout=dropout)
        self.swin_3 = Swin_Model(128, 128, dim_head, depth, 1, 8, (32, 32), 4, dropout=dropout)
        self.conv_pred1 = nn.Conv2d(64, output_nc, kernel_size=(3, 3), padding=1)
        self.conv_pred2 = nn.Conv2d(128, output_nc, kernel_size=(3, 3), padding=1)
        self.conv_pred3 = nn.Conv2d(256, output_nc, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x_2 = self.swin_1(x)
        x_4 = self.swin_2(x_2)
        x_8 = self.swin_3(x_4)

        x_2 = self.conv_pred1(x_2)
        x_4 = self.conv_pred2(x_4)
        x_8 = self.conv_pred3(x_8)

        return x_2, x_4, x_8


class Swin_block1(nn.Module):
    def __init__(self,
                 chan,
                 if_patch_embading,
                 dim,
                 dim_head,
                 depth,
                 num_heads,
                 input_res,
                 window_size,
                 patch_size,
                 qkv_bias=None,
                 dropout=0.0
                 ):
        super().__init__()

        self.if_patch_embading = if_patch_embading
        self.depth = depth
        self.patch_embed = Patch_Embeding(chan, dim=dim, patch_size=patch_size)
        self.W_MSA_block = Swin_Block(dim, dim_head, num_heads, input_res, window_size, dropout, qkv_bias, shift_size=0)
        self.SW_MSA_block = Swin_Block(dim, dim_head, num_heads, input_res, window_size, dropout, qkv_bias,
                                       shift_size=window_size // 2)
        self.patch_merging = Patch_Merging(dim // 2)

    def forward(self, x):
        if self.if_patch_embading:
        #     # x = self.patch_merging(x)
            x = self.patch_embed(x)
        else:
            x = x.flatten(2).transpose(1, 2)
        for i in range(self.depth):
            x = self.W_MSA_block(x)
            x = self.SW_MSA_block(x)

        h, w = int(np.sqrt(x.size(1))), int(np.sqrt(x.size(1)))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(x.size(0), x.size(1), h, w)

        return x


class Swin_Model1(nn.Module):
    def __init__(self,
                 chan,
                 output_nc,
                 depth,
                 patch_size,
                 dropout
                 ):
        super().__init__()

        self.patch_size = patch_size
        self.depth = depth
        self.patch_embed = Patch_Embeding(chan, 64, patch_size=patch_size)
        self.swin_block1 = Swin_block1(3, True, 64, 8, 2, 8, (64, 64), 8, 4, dropout=dropout)
        self.swin_block2 = Swin_block1(64, True, 128, 8, 2, 8, (32, 32), 8, 2, dropout=dropout)
        self.swin_block3 = Swin_block1(128, True, 256, 8, 2, 8, (16, 16), 8, 2, dropout=dropout)

        self.conv_pred1 = nn.Conv2d(64, output_nc, kernel_size=(3, 3), padding=1)
        self.conv_pred2 = nn.Conv2d(128, output_nc, kernel_size=(3, 3), padding=1)
        self.conv_pred3 = nn.Conv2d(256, output_nc, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        # x_4 = self.patch_embed(x)
        x_4 = self.swin_block1(x)
        x_8 = self.swin_block2(x_4)
        x_16 = self.swin_block3(x_8)

        x_4 = self.conv_pred1(x_4)
        x_8 = self.conv_pred2(x_8)
        x_16 = self.conv_pred3(x_16)

        return x_4, x_8, x_16


class Swin_Model2(nn.Module):
    def __init__(self,
                 chan,
                 dim,
                 patch_size,
                 num_heads,
                 input_res,
                 window_size,
                 depth,
                 qkv_bias=None
                 ):
        super().__init__()

        self.patch_size = patch_size
        self.depth = depth
        self.patch_embed = Patch_Embeding(chan, dim, patch_size=patch_size)
        self.W_MSA_block = Swin_Block(dim, num_heads, input_res, window_size, qkv_bias, shift_size=0)
        self.SW_MSA_block = Swin_Block(dim, num_heads, input_res, window_size, qkv_bias, shift_size=window_size // 2)
        self.patch_merging = Patch_Merging(input_res, dim)

    def forward(self, x):
        H, W = self.resolution
        B, _, C = x.shape
        x = self.patch_embed(x)
        for i in range(self.depth):
            x = self.W_MSA_block(x)
            x = self.SW_MSA_block(x)
        # out = self.patch_merging(x)
        out = x.reshape(B, H, W, C)

        # h, w = int(np.sqrt(out.size(1))), int(np.sqrt(out.size(1)))
        # x = out.permute(0, 2, 1)
        # x = x.contiguous().view(out.size(0), out.size(2), h, w)

        return out
