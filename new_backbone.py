import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

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
        # return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], 1).values
        out = self.sigmoid(self.conv(out))
        return out * x

class MBAF_Block(nn.Module):
    def __init__(self, in_channel, branch=(1, 3, 5)):
        super(MBAF_Block, self).__init__()

        branches, local_att = [], []

        for b in branch:
            branches.append(nn.Sequential(
                nn.Conv3d(in_channel, in_channel * 2, kernel_size=b, padding=(b - 1) // 2),
                nn.BatchNorm3d(in_channel * 2),
                nn.ReLU(),
                nn.Conv3d(in_channel * 2, in_channel, kernel_size=1),
                nn.BatchNorm3d(in_channel),
                nn.ReLU()
            ))
            local_att.append(nn.Sequential(
                nn.Conv3d(in_channel, in_channel, kernel_size=1),
                nn.Sigmoid()
            ))
        self.branches = nn.ModuleList(branches)
        self.local_att = nn.ModuleList(local_att)
        self.reduce_channel = nn.Sequential(nn.Conv3d(in_channel * len(branch), in_channel, 1),
                                            nn.BatchNorm3d(in_channel), nn.ReLU())

    def forward(self, x):
        block_out = []
        for i in range(len(self.branches)):
            block = self.branches[i]
            local = self.local_att[i]
            b = block(x)
            y = b * local(b)
            block_out.append(y)
        block_all = torch.cat(block_out, 1)
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
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True) if layer_scale_init_value > 0 else None
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
    """

    def __init__(
            self,
            in_chans=1,
            dims=None,
            layer_scale_init_value=1e-6,
    ):
        super().__init__()
        self.DS1 = nn.Sequential(
            nn.Conv3d(in_chans, 32, kernel_size=4, stride=2, padding=1),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            LayerNorm3D(64, eps=1e-6, data_format="channels_first"), )

        self.DS2 = nn.Sequential(
            nn.Conv3d(64, dims[1], kernel_size=2, stride=2),
            LayerNorm3D(dims[1], eps=1e-6, data_format="channels_first"), )

        self.DS3 = nn.Sequential(
            nn.Conv3d(dims[1], dims[1], kernel_size=2, stride=2),
            LayerNorm3D(dims[1], eps=1e-6, data_format="channels_first"), )

        self.DS4 = nn.Sequential(
        nn.Conv3d(dims[1], dims[2], kernel_size=2, stride=2),
        LayerNorm3D(dims[2], eps=1e-6, data_format="channels_first"),
        )
        self.conv_block1 = Block3D(
            dim=64,
            layer_scale_init_value=layer_scale_init_value)
        self.conv_block2 = Block3D(
            dim=dims[1],
            layer_scale_init_value=layer_scale_init_value)
        self.conv_block3 = Block3D(
            dim=dims[1],
            layer_scale_init_value=layer_scale_init_value)

        self.MS1 = MBAF_Block(64)
        self.MS2 = MBAF_Block(128, [1, 3])

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_block1(self.MS1(self.DS1(x)))
        # x = self.DS1(x)
        # x = self.MS1(x)
        # x = self.conv_block1(x)
        x = self.conv_block2(self.MS2(self.DS2(x)))
        # x = self.DS2(x)
        # x = self.MS2(x)
        # x = self.conv_block2(x)
        x = self.conv_block3(self.DS3(x))
        # x = self.DS3(x)
        # x = self.conv_block3(x)

        x = self.DS4(x)
        x = x.flatten(2).transpose(1, 2)
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
    model = ConvNeXt3D(in_chans=1, dims=[64, 128, 768])  # [3, 3, 9, 3]
    return model


if __name__ == "__main__":
    input = torch.rand((1, 1, 112, 112, 112)).cuda()
    net = convnext_tiny_3d()
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
    print(net)
