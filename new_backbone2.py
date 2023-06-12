import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


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


def DropKey(Q, K, V, use_Dropkey, mask_radio):
    attn = (Q * (Q.shape[1] ** -0.5)) @ K.transpose(-2, -1)
    if use_Dropkey == True:
        m_r = torch.ones_like(attn) * mask_radio
        attn = attn + torch.bernoulli(m_r) * -1e12
    attn = attn.softmax(dim=-1)
    x = attn @ V
    return x


class Block3D(nn.Module):
    r"""3D ConvNeXt Block. Similar to the 2D version, but with 3D convolutions.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., i=1, layer_scale_init_value=1e-6):
        super().__init__()
        self.i = i
        self.dwconv0 = nn.Conv3d(dim, self.i * dim, kernel_size=1, groups=dim)
        self.dwconv = nn.Conv3d(self.i * dim, self.i * dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm3D(self.i * dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.i * dim, 4 * self.i * dim)  # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * self.i * dim, self.i * dim)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((self.i * dim)),
                requires_grad=True,
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if self.i != 1:
            x = self.dwconv0(x)
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
            i=1,
            layer_scale_init_value=layer_scale_init_value)

        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.DS1(x)
        x = self.conv_block1(x)
        x = self.SA1(x)
        x = self.DS2(x)
        x = self.conv_block2(x)
        x = self.SA2(x)
        x = self.DS3(x)
        x = self.conv_block3(x)
        x = self.DS4(x)
        x = self.conv_block4(x)

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
    model = ConvNeXt3D(in_chans=1, dims=[32, 64, 128, 768])  # [3, 3, 9, 3]
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
