import torch
from torch import nn
from typing import Tuple, Union, List, Any

__all__ = [
    "ConvBlock",
    "MBConvBlock",
    "Classifier",
]

Normalization = nn.BatchNorm2d
Activation = nn.SiLU


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ):
        super().__init__()
        layer = []
        layer += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kwargs.get("kernel_size"),
                stride=kwargs.get("stride", 1),
                padding=kwargs.get("padding", 0),
                groups=kwargs.get("groups", 1),
                bias=kwargs.get("bias", False),
            )
        ]
        layer += [Normalization(out_channels)]
        if kwargs.get("act", True):
            layer += [Activation()]

        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SE(nn.Module):
    # SE0.25
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        r: int = 4,
    ) -> None:
        super().__init__()
        dim = in_channels // r
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(out_channels, dim),
            Activation(),
            nn.Linear(dim, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.size()
        sc = x
        x = self.squeeze(x)
        x = x.view(B, -1)
        x = self.excitation(x)
        x = x.view(B, C, 1, 1)
        return sc * x


class DepthWiseConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        out_channels = in_channels
        layer = []
        layer += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_channels,
                padding=kwargs.get("padding", 0),
                bias=kwargs.get("bias", False),
            )
        ]
        layer += [Normalization(out_channels)]
        if kwargs.get("act", True):
            layer += [Activation()]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        is_fused: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and in_channels == out_channels

        dim = in_channels * expand_ratio
        layer = []
        if is_fused:
            layer += [
                ConvBlock(in_channels, dim, kernel_size=3, stride=stride, padding=1),
            ]
        else:
            layer += [
                ConvBlock(in_channels, dim, kernel_size=1, stride=1),
                DepthWiseConvBlock(dim, 3, stride, padding=1),
                SE(in_channels, dim),
            ]
        layer += [
            ConvBlock(dim, out_channels, kernel_size=1, stride=1, act=False),
        ]
        self.net = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return x + self.net(x)
        else:
            return self.net(x)


class Classifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)
