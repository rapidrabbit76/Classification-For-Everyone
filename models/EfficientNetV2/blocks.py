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


class ConvBlock(nn.Module):
    def __init__(
        self, inp: int, outp: int, k: int, s: int = 1, p: int = 0, act: bool = True
    ):
        super().__init__()
        layer = [nn.Conv2d(inp, outp, k, s, p, bias=False)]
        layer += [Normalization(outp)]
        if act:
            layer += [Activation()]

        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SE(nn.Module):
    # SE0.25 => r:4
    def __init__(self, inp: int, outp: int, r: int = 4) -> None:
        super().__init__()
        dim = inp // r
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(outp, dim),
            Activation(),
            nn.Linear(dim, outp),
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
        self, inp: int, k: int, s: int = 1, p: int = 0, act: bool = True
    ) -> None:
        super().__init__()
        outp = inp
        layer = [nn.Conv2d(inp, outp, k, s, p, groups=inp, bias=False)]
        layer += [Normalization(outp)]
        if act:
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
        self.identity = stride == 1 and in_channels == out_channels

        dim = in_channels * expand_ratio
        layer = [
            ConvBlock(in_channels, dim, 1, 1),
            DepthWiseConvBlock(dim, 3, stride, 1),
            SE(in_channels, dim),
        ]

        if is_fused:
            layer = [ConvBlock(in_channels, dim, 3, stride, 1)]

        layer += [ConvBlock(dim, out_channels, 1)]
        self.net = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.net(x)
        return x + net if self.identity else net


class Classifier(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(inp, outp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)
