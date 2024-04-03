import torch.nn as nn


def down_conv(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False)


def up_conv(in_planes, out_planes):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
