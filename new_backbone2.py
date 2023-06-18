import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        super(eca_block, self).__init__()

        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, d, h, w = inputs.shape
        x = self.conv(self.avg_pool(inputs).view([b, 1, c])).view([b, c, 1, 1, 1])
        x = self.sigmoid(x)
        outputs = x * inputs
        return outputs


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout = torch.max(x, dim=1, keepdim=True).values
        out = torch.cat([avgout, maxout], 1)
        out = self.sigmoid(self.conv(out))
        return out * x


class ChannelAttention(nn.Module):
    def __init__(self, c_in, m_in):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Linear(c_in, m_in)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(m_in, c_in)
        self.flatten = nn.Flatten()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        avg_x = torch.reshape(avg_x, [avg_x.shape[0], -1])
        max_x = torch.reshape(max_x, [max_x.shape[0], -1])
        avg_out = self.fc2(self.relu1(self.fc1(avg_x)))
        max_out = self.fc2(self.relu1(self.fc1(max_x)))
        out = (avg_out + max_out).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return self.sigmoid(out)


class CDAF_Block(nn.Module):
    def __init__(self, in_channel):
        super(CDAF_Block, self).__init__()
        # cross dimension
        self.conv_bcxyz = nn.Sequential(nn.Conv3d(in_channel, in_channel * 2, kernel_size=3, padding=1),
                                        nn.BatchNorm3d(in_channel * 2), nn.ReLU(),
                                        nn.Conv3d(in_channel * 2, in_channel, 1), nn.BatchNorm3d(in_channel), nn.ReLU())
        self.conv_bcyzx = nn.Sequential(nn.Conv3d(in_channel, in_channel * 2, kernel_size=3, padding=1),
                                        nn.BatchNorm3d(in_channel * 2), nn.ReLU(),
                                        nn.Conv3d(in_channel * 2, in_channel, 1), nn.BatchNorm3d(in_channel), nn.ReLU())
        self.conv_bcxzy = nn.Sequential(nn.Conv3d(in_channel, in_channel * 2, kernel_size=3, padding=1),
                                        nn.BatchNorm3d(in_channel * 2), nn.ReLU(),
                                        nn.Conv3d(in_channel * 2, in_channel, 1), nn.BatchNorm3d(in_channel), nn.ReLU())
        self.reduce_channel = nn.Sequential(nn.Conv3d(in_channel * 3, in_channel, 1), nn.BatchNorm3d(in_channel),
                                            nn.ReLU())
        self.sp_xyz = SpatialAttention()
        self.sp_yzx = SpatialAttention()
        self.sp_xzy = SpatialAttention()

        self.eca1 = eca_block(in_channel)
        self.eca2 = eca_block(in_channel)
        self.eca3 = eca_block(in_channel)

    def forward(self, x):  # pass downsample, from size(Bx1x121x145x121) to size(Bx128x30x36x30)
        # x1 = self.sp_xyz(self.eca1(self.conv_bcxyz(x)))
        # x2 = self.sp_yzx(self.eca2(self.conv_bcyzx(x.permute([0, 1, 3, 4, 2])))).permute([0, 1, 4, 2, 3])
        # x3 = self.sp_xzy(self.eca3(self.conv_bcxzy(x.permute([0, 1, 2, 4, 3])))).permute([0, 1, 2, 4, 3])
        x1 = self.sp_xyz(self.conv_bcxyz(x))
        x2 = self.sp_yzx(self.conv_bcyzx(x.permute([0, 1, 3, 4, 2]))).permute([0, 1, 4, 2, 3])
        x3 = self.sp_xzy(self.conv_bcxzy(x.permute([0, 1, 2, 4, 3]))).permute([0, 1, 2, 4, 3])
        block_all = torch.cat([x1, x2, x3], 1)
        return self.reduce_channel(block_all)


class Block3D(nn.Module):
    r"""3D ConvNeXt Block. Similar to the 2D version, but with 3D convolutions.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm3D(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((dim)),
                requires_grad=True,
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, D, H, W, C) -> (N, C, D, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXt3D(nn.Module):
    r"""ConvNeXt 3D model.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self,
            in_chans=1,
            dims=None,
            layer_scale_init_value=1e-6,
    ):
        super().__init__()

        if dims is None:
            dims = []
        self.DS1 = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=2, padding=1),
            # nn.Conv3d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            LayerNorm3D(dims[0], eps=1e-6, data_format="channels_first"),
        )

        self.DS2 = nn.Sequential(
            nn.Conv3d(dims[0], dims[1], kernel_size=2, stride=2),
            LayerNorm3D(dims[1], eps=1e-6, data_format="channels_first"), )

        self.DS3 = nn.Sequential(
            nn.Conv3d(dims[1], dims[2], kernel_size=2, stride=2),
            LayerNorm3D(dims[2], eps=1e-6, data_format="channels_first"), )

        self.DS4 = nn.Sequential(
            nn.Conv3d(dims[2], dims[3], kernel_size=2, stride=2),
            LayerNorm3D(dims[3], eps=1e-6, data_format="channels_first"),
        )
        self.DS5 = nn.Sequential(
            nn.Conv3d(dims[3], dims[4], kernel_size=2, stride=1),
            LayerNorm3D(dims[4], eps=1e-6, data_format="channels_first"),
        )
        self.conv_block1 = Block3D(
            dim=dims[0],
            layer_scale_init_value=layer_scale_init_value)
        self.conv_block2 = Block3D(
            dim=dims[1],
            layer_scale_init_value=layer_scale_init_value)
        self.conv_block3 = Block3D(
            dim=dims[2],
            layer_scale_init_value=layer_scale_init_value)
        self.conv_block4 = Block3D(
            dim=dims[3],
            layer_scale_init_value=layer_scale_init_value)

        # self.SA1 = SpatialAttention()
        # self.SA2 = SpatialAttention()

        self.MS = CDAF_Block(64)
        # self.MS1 = CDAF_Block(256)

        # self.eca = eca_block(768)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.DS1(x)
        x = self.conv_block1(x)

        x = self.DS2(x)
        x = self.conv_block2(x)

        x = self.MS(x)

        x = self.DS3(x)
        x = self.conv_block3(x)

        x = self.DS4(x)
        x = self.conv_block4(x)
        # x = self.MS1(x)

        x = self.DS5(x)

        x = x.flatten(2).transpose(1, 2)
        # x = self.head(x)
        return x


class LayerNorm3D(nn.Module):
    r"""LayerNorm that supports three-dimensional inputs.

    Args:
        normalized_shape (int or tuple): Input shape from an expected input. If it is a single integer,
            it is treated as a singleton tuple. Default: 1
        eps (float): A value added to the denominator for numerical stability. Default: 1e-6
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            a = self.weight.shape
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


@register_model
def convnext_tiny_3d():
    model = ConvNeXt3D(in_chans=1, dims=[32, 64, 128, 256, 576])  # [3, 3, 9, 3]
    return model


if __name__ == "__main__":
    input = torch.rand((1, 1, 112, 112, 112)).cuda()
    # input = torch.rand((1, 32, 56, 56, 56)).cuda()
    net = convnext_tiny_3d()
    # net = eca_block(768)
    # net = CDAF_Block(32)
    net.cuda()
    param_size = 0
    param_sum = 0
    for param in net.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    print("{:.3f} MB".format(torch.cuda.memory_allocated(0) / 1024 / 1024))
    out = net(input)
    print(out.shape)
    # print(net)
