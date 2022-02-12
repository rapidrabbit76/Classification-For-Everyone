import torch
from torch import nn
from typing import Tuple, Union, List, Any

__all__ = [
    "ConvBlock",
    "ShuffleNetUnit",
    "Classifier",
]

Normalization = nn.BatchNorm2d
Activation = nn.SiLU


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
    def __init__(self, dim: int, r: int) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(dim, dim // r),
            Activation(),
            nn.Linear(dim // r, dim),
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
        is_fused: bool,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        assert stride in [1, 2]
        dim = in_channels * expand_ratio
        layer = []
        self.identity = stride == 1 and in_channels == out_channels
        if is_fused:
            layer += [
                ConvBlock(in_channels, dim, kernel_size=3, stride=stride),
            ]
        else:
            layer += [
                ConvBlock(in_channels, dim, kernel_size=1, stride=1),
                DepthWiseConvBlock(dim, 3, stride, padding=1),
                SE(dim, expand_ratio),
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
