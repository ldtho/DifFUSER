import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

class DoubleInputSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.seq = nn.Sequential(*args)

    def forward(self, x, c):
        for module in self.seq:
            x, c = module(x, c)
        return x, c


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """
    def __init__(self, in_channels,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm=False,
                 bias=False,
                 activation=False,
                 onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding,
                                        groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                        stride=1, padding=0)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.1, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.act = nn.SiLU()
            # self.swish = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.act(x)

        return x


class Modulate(nn.Module):
    """
    Simple example of a 'modulate' block that can incorporate shift, scale, or gating.
    """
    def __init__(self, shift_scale_gate='shift_scale_gate'):
        super(Modulate, self).__init__()
        self.shift_scale_gate = shift_scale_gate

    @force_fp32()
    def forward(self, x, gate, shift, scale):
        # example gating logic
        if 'scale' in self.shift_scale_gate:
            x = x * (scale + 1)
        if 'shift' in self.shift_scale_gate:
            x = x + shift
        if 'gate' in self.shift_scale_gate:
            x = torch.sigmoid(gate) * x

        return x


class AttentionBlock(nn.Module):
    """
    Custom block that does 'attention-like' or feature fusion logic.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 first_layer=False,
                 shift_scale_gate='shift_scale_gate',
                 **block_kwargs):
        super().__init__()
        self.first_block = first_layer
        self.modulate = Modulate(shift_scale_gate)

        if first_layer:
            # example logic if first layer requires special downsampling
            self.x_down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            )
            self.c_down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
                nn.BatchNorm2d(out_channels, eps=1e-3),
                nn.SiLU(),
            )
        else:
            self.x_down_sample = None
            self.c_down_sample = None

        self.norm1 = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channels, 3 * out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.SiLU(),
        )
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, c):
        # If this is the first block, special downsampling of x and c
        if self.first_block:
            x = self.x_down_sample(x)
            c = self.c_down_sample(c)

        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        h = self.modulate(self.norm1(x), gate_mlp, shift_mlp, scale_mlp)
        h = F.silu(h)
        h = self.block(h)
        x = h + self.conv1(F.silu(x))
        return x, c


class SeparableConvBlock(nn.Module):
    """
    Depthwise + Pointwise Convolution block, commonly used in lightweight networks.
    """
    def __init__(self, in_channels,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm=False,
                 bias=False,
                 activation=False,
                 onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding,
                                        groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                        stride=1, padding=0)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.1, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.act(x)

        return x


class DoubleInputSequential(nn.Module):
    """
    A small wrapper to sequentially apply blocks that each return (x, c).
    """
    def __init__(self, *args):
        super().__init__()
        self.seq = nn.Sequential(*args)

    def forward(self, x, c):
        for module in self.seq:
            x, c = module(x, c)
        return x, c
