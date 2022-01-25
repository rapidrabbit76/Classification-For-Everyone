from operator import mod
import torch
from torch import nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        return self.conv(x)


class TransitionBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return self.pool(x)


class BottleNeckBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
    ) -> None:
        super().__init__()
        dim = growth_rate * 4
        self.residual = nn.Sequential(
            ConvBlock(in_channels, dim, 1, 1, 0),
            ConvBlock(dim, growth_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.residual(x)], dim=1)


class DenseBlock(nn.Module):

    def __init__(
        self,
        blocks: int,
        dim: int,
        growth_rate: int,
    ) -> None:
        super().__init__()
        layers = []
        for _ in range(blocks):
            layers += [BottleNeckBlock(dim, growth_rate)]
            dim += growth_rate
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
