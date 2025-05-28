from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional
from models.base_block import *
import torch.nn.functional as F
import warnings
from einops import rearrange
import torchvision.models as models
from models.swin_block import Swin_Block
from timm.models.swin_transformer import SwinTransformerBlock
from models.help_funcs import  TransformerDecoder

# （双分支）特征提取器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        for n, m in self.resnet.layer2.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.resnet.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(64+128, 128, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(512+256, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        # initialize_weights(self.conv)

    def forward(self, x):
        x0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))        #1/2
        xm = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(xm)  #1/4
        x2 = self.resnet.layer2(x1)  #1/4
        x3 = self.resnet.layer3(x2)  #1/8
        x4 = self.resnet.layer4(x3)#1/8
        x2 = torch.cat([x1, x2], 1)
        x4 = torch.cat([x3, x4], 1)
        x2 = self.conv1(x2)
        x4 = self.conv2(x4)
        return x0,  x2,  x4

# 轴向注意力基础模块
class AxialAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super(AxialAttention, self).__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim* 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x, axis='height'):
        """
        x: 输入特征图 [B, C, H, W]
        axis: 注意力轴向 ('height'或'width')
        """
        b, c, h, w = x.shape

        # 生成QKV投影
        qkv = self.to_qkv(x)  # [B, 3*heads*dim_head, H, W]
        q, k, v = rearrange(qkv, 'b (qkv h d) x y -> qkv b h (x y) d',
                            qkv=3, h=self.heads)  # 分解为Q/K/V

        # 轴向注意力计算
        if axis == 'width':
            q, k, v = map(lambda t: rearrange(t, 'b h (x y) d -> b h y x d', x=h, y=w, h = self.heads), (q, k, v))
        elif axis == 'height':
            q, k, v = map(lambda t: rearrange(t, 'b h (x y) d -> b h x y d', x=h, y=w, h = self.heads), (q, k, v))

        # 注意力分数计算
        dots = torch.einsum('b h x i d, b h x j d -> b h x i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # 特征聚合
        out = torch.einsum('b h x i j, b h x j d -> b h x i d', attn, v)

        # 恢复空间维度
        if axis == 'width':
            out = rearrange(out, 'b h y x d -> b h (x y) d', y=w, x=h, h = self.heads)
        elif axis == 'height':
            out = rearrange(out, 'b h x y d -> b h (x y) d', x=h, y=w, h = self.heads)

        # 合并多头输出
        out = rearrange(out, 'b h n d -> b (h d) n').view(b, -1, h, w)
        return self.to_out(out)

# 双轴向注意力模块
class AxialAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super(AxialAttentionBlock, self).__init__()
        # 行注意力（水平轴）
        self.height_att = AxialAttention(dim, heads, dim_head)
        # 列注意力（垂直轴）
        self.width_att = AxialAttention(dim, heads, dim_head)

        # 归一化层
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        # 残差连接1
        res = x

        # 水平轴注意力
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.height_att(x, axis='height')

        x = x + res

        # 残差连接2
        res = x
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # 垂直轴注意力
        x = self.width_att(x, axis='width')

        x = x + res

        return  self.ffn(x)+x

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

# MLP
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 空间缩减注意力
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # bs, 16384, 32 => bs, 16384, 32 => bs, 16384, 8, 4 => bs, 8, 16384, 4
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # bs, 16384, 32 => bs, 32, 128, 128
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # bs, 32, 128, 128 => bs, 32, 16, 16 => bs, 256, 32
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            # bs, 256, 32 => bs, 256, 64 => bs, 256, 2, 8, 4 => 2, bs, 8, 256, 4
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # bs, 8, 16384, 4 @ bs, 8, 4, 256 => bs, 8, 16384, 256
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # bs, 8, 16384, 256  @ bs, 8, 256, 4 => bs, 8, 16384, 4 => bs, 16384, 32
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # bs, 16384, 32 => bs, 16384, 32
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 语义增强和轴向注意模块
class Block(nn.Module):

    def __init__(self, dim, num_heads, window_size, depth, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.depth =depth
        self.swin_block = Swin_Block(dim, num_heads, window_size, 8, attn_drop=attn_drop, drop_path=drop_path)
        self.axial_block = AxialAttentionBlock(dim, num_heads, dim//num_heads)
        # self.patch_embed = PatchEmbeding(dim, dim, patch_size, dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        for i in range(self.depth):
            res = x
            # x = self.patch_embed(x)
            B, C, H, W = x.shape
            # 重塑
            x = x.flatten(2)
            x = x.transpose(-1, -2)
            B, _, C = x.shape
            # 基于窗口多头注意力的SET局部编码器
            x = self.swin_block(x, H, W)
            # 基于空间缩减注意力的SET全局编码器
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            x = x.transpose(1, 2).view(B, C, H, W)
            # 轴向注意力模块
            res = self.axial_block(res)
            x = x + res
        return x

# 三维卷积融合
class cross_fuse3_3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cross_fuse3_3d, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=[3, 3, 3], stride=1, padding=1, bias=True),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x1, x2, x3):
        tensor1 = x1
        tensor2 = x2
        tensor3 = x3
        b, c, h, w = tensor1.shape
        tensor1 = tensor1.view(b, c, h * w)
        tensor2 = tensor2.view(b, c, h * w)
        tensor3 = tensor3.view(b, c, h * w)
        cross_x = torch.cat((tensor1, tensor2, tensor3), dim=2)
        cross_x = cross_x.view(b, 3*c, h, w)
        cross_x = cross_x.unsqueeze(1)
        cross_x = self.conv3d(cross_x)
        cross_x = cross_x.squeeze(1)
        cross_x = self.fuse_conv(cross_x)
        cross_x = self.dropout(cross_x)
        cross_x = self.fuse_conv1(cross_x)
        cross_x = self.dropout(cross_x)
        return cross_x

# 变化信息提取模块
class DFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(2*dim, dim, 1, 1)
        self.cross_fuse_c1 = cross_fuse3_3d(in_channels=3 * dim, out_channels= dim)

    def forward(self, x1, x2):
        x_diff = torch.abs(x1-x2)
        x_add = x1+x2
        x_cat = self.conv1(torch.cat([x1,  x2],1))
        f_cd = self.cross_fuse_c1(x_cat, x_add, x_diff)

        return f_cd

# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)*x

# 基于语义增强和轴向注意的语义分支解码器
class ss_Decoder(nn.Module):
    def __init__(self, dim, embedding_dim, depth, attn_drop, drop_path):
        super().__init__()
        # self.transformer_block1 = Block(dim[3], 4, 4, 8, sr_ratio=4, attn_drop=attn_drop, drop_path=drop_path)
        self.transformer_block2 = Block(dim[2], 4, 4, depth, 4, sr_ratio=8, attn_drop=attn_drop, drop_path=drop_path)
        self.transformer_block3 = Block(dim[1], 2, 4, depth, 4, sr_ratio=16, attn_drop=attn_drop, drop_path=drop_path)
        self.fuse2 = nn.Sequential(nn.Conv2d(dim[2]+dim[1], dim[1], 3,1 ,2,2),
            nn.BatchNorm2d(dim[1]),  # 必加
            nn.ReLU(inplace=True),  # 可选
            nn.Dropout(0.)  # 遥感任务可追加Dropout
        )
        self.fuse3 = nn.Sequential(nn.Conv2d(dim[1]+dim[0], embedding_dim, 3,1 ,2,2),
            nn.BatchNorm2d(embedding_dim),  # 必加
            nn.ReLU(inplace=True),  # 可选
            nn.Dropout(0.)  # 遥感任务可追加Dropout
        )
        self.CA1 = ChannelAttentionModule(dim[2])
        self.CA2 = ChannelAttentionModule(dim[1])
        self.CA3 = ChannelAttentionModule(embedding_dim)

    def forward(self, x1, x2, x3):
        d3 = self.transformer_block2(x3)
        # 上采样融合
        d3_up = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.CA2(self.fuse2(torch.cat([d3_up, x2],1)))

        d2 = self.transformer_block3(d2)

        d2_up = F.interpolate(d2,scale_factor=2,  mode='bilinear', align_corners=False)
        d1 = self.CA3(self.fuse3(torch.cat([d2_up, x1],1)))

        return d1, d2, d3

# 通道特征交换模块
def channels_exchange(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    交换两个张量中所有奇数索引的通道（索引从0开始计算）

    参数:
        tensor1 (torch.Tensor): 形状为 (B, C, H, W) 的张量
        tensor2 (torch.Tensor): 形状为 (B, C, H, W) 的张量

    返回:
        (torch.Tensor, torch.Tensor): 交换后的新张量
    """
    assert tensor1.shape == tensor2.shape, "两个张量形状必须一致"
    C = tensor1.size(1)

    # 生成奇数索引列表（例如C=5时，索引1,3）
    odd_indices = [i for i in range(C) if i % 2 == 1]
    if len(odd_indices) == 0:
        return tensor1.clone(), tensor2.clone()  # 无奇数通道可交换

    # 创建副本避免修改原始张量
    new_tensor1 = tensor1.clone()
    new_tensor2 = tensor2.clone()

    # 交换奇数索引的通道
    new_tensor1[:, odd_indices] = tensor2[:, odd_indices]
    new_tensor2[:, odd_indices] = tensor1[:, odd_indices]

    return new_tensor1, new_tensor2

# SET_3DC方法
class swin_T_3D(nn.Module):
    def __init__(self, input_nc=3, num_classes=2, dim = [64, 128, 256], decoder_softmax=False, embed_dim=128):
        super(swin_T_3D, self).__init__()
        self.backbone = Encoder()
        self.DFM1 = DFM(embed_dim)
        # self.DFM2 = DFM(dim[1])
        # self.DFM3 = DFM(dim[2])
        # self.DFM4 = DFM(dim[3])
        self.ss_Decoder = ss_Decoder(dim, embed_dim, 1,  attn_drop=0.1, drop_path=0.1)
        # self.cd_Decoder = cd_Decoder(dim, embed_dim)

        self.ss_classifier1 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim//2, kernel_size=1), nn.BatchNorm2d(embed_dim//2), nn.ReLU(),
                                             nn.Conv2d(embed_dim//2, num_classes, kernel_size=1))

        self.cd_classifier = nn.Sequential(nn.Conv2d(embed_dim, embed_dim//2, kernel_size=1), nn.BatchNorm2d(embed_dim//2), nn.ReLU(),
                                           nn.Conv2d(embed_dim//2, 1, kernel_size=1))

    def forward(self, x1, x2):
        x_size = x1.size()
        # 双分支特征提取器
        c1_1, c2_1, c3_1 = self.backbone(x1)
        c1_2, c2_2, c3_2 = self.backbone(x2)
        # 通道特征交换模块
        c3_1, c3_2 = channels_exchange(c3_1, c3_2)
        c2_1, c2_2 = channels_exchange(c2_1, c2_2)
        c1_1, c1_2 = channels_exchange(c1_1, c1_2)

        # 基于语义增强和轴向注意的语义分支解码器
        s1_1, s2_1, s3_1 = self.ss_Decoder(c1_1, c2_1, c3_1)
        s1_2, s2_2, s3_2 = self.ss_Decoder(c1_2, c2_2, c3_2)

        # 变化信息提取模块
        cd1 = self.DFM1(s1_1, s1_2)

        # 语义分割分类器
        s1 = self.ss_classifier1(s1_1+cd1)
        s2 = self.ss_classifier1(s1_2+cd1)
        # 变化检测分类器
        p_bcd = self.cd_classifier(cd1)

        return F.upsample(p_bcd, x_size[2:], mode='bilinear'), \
               F.upsample(s1, x_size[2:], mode='bilinear'), \
               F.upsample(s2, x_size[2:], mode='bilinear')