from operator import mod
import torch
from torch import nn


Normalization = nn.BatchNorm2d
Activation = nn.ReLU


class ConvBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            Normalization(inp),
            Activation(inplace=True),
            nn.Conv2d(inp, outp, k, s, p, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransitionBlock(nn.Module):
    def __init__(self, inp: int, outp: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Normalization(inp),
            Activation(inplace=True),
            nn.Conv2d(inp, outp, 1, 1, 0, bias=False),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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
