import torch
from torch import nn

from typing import Tuple, Union, List

_size_2_t = Union[int, Tuple[int, int]]


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: int = 1,
        norm: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(in_channels) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class Inceptionx3(nn.Module):

    def __init__(
        self,
        in_channels: int,
        dims: List[int],
        norm: bool,
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, dims[0], 1, stride=1, padding=0, norm=norm),
            ConvBlock(dims[0], dims[0], 3, stride=1, padding=1, norm=norm),
            ConvBlock(dims[0], dims[0], 3, stride=1, padding=1, norm=norm),
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, dims[1], 1, stride=1, padding=0, norm=norm),
            ConvBlock(dims[1], dims[1], 3, stride=1, padding=1, norm=norm),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, dims[2], 1, stride=1, norm=norm),
        )
        self.branch4 = ConvBlock(
            in_channels,
            dims[3],
            kernel_size=1,
            stride=1,
            padding=0,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class Inceptionx5(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # TODO


class Inceptionx2(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # TODO