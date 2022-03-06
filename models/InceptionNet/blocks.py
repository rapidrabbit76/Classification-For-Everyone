from typing import *

import torch
from torch import Tensor, nn

_size_2_t = Union[int, Tuple[int, int]]

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


class ConvBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        k: _size_2_t = 3,
        s: _size_2_t = 1,
        p: _size_2_t = 1,
    ):
        super().__init__()
        layer = [nn.Conv2d(inp, outp, k, s, p, bias=False)]

        layer += [Normalization(outp, eps=0.001)]
        layer += [Activation()]
        self.block = nn.Sequential(*layer)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Inceptionx3(nn.Module):
    def __init__(self, inp: int, dims: List[int]) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(inp, dims[0], 1, 1, 0),
            ConvBlock(dims[0], dims[0], 3, 1, 1),
            ConvBlock(dims[0], dims[0], 3, 1, 1),
        )

        self.branch2 = nn.Sequential(
            ConvBlock(inp, dims[1], 1, 1, 0),
            ConvBlock(dims[1], dims[1], 3, 1, 1),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            ConvBlock(inp, dims[2], 1, 1, 0),
        )
        self.branch4 = ConvBlock(inp, dims[3], 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class Inceptionx5(nn.Module):
    def __init__(
        self,
        inp: int,
        dims: List[int],
        n: int = 7,
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(inp, dims[0], 1, 1, 0),
            ConvBlock(dims[0], dims[0], (1, n), 1, (0, n // 2)),
            ConvBlock(dims[0], dims[0], (n, 1), 1, (n // 2, 0)),
            ConvBlock(dims[0], dims[0], (1, n), 1, (0, n // 2)),
            ConvBlock(dims[0], dims[0], (n, 1), 1, (n // 2, 0)),
        )

        self.branch2 = nn.Sequential(
            ConvBlock(inp, dims[1], 1, 1, 0),
            ConvBlock(dims[1], dims[1], (1, n), 1, (0, n // 2)),
            ConvBlock(dims[1], dims[1], (n, 1), 1, (n // 2, 0)),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            ConvBlock(inp, dims[2], 1, 1, 0),
        )

        self.branch4 = ConvBlock(inp, dims[3], 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class Inceptionx2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dims: List[int],
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, dims[0] // 4, 1, 1, 0),
            ConvBlock(dims[0] // 4, dims[0] // 4, 3, 1, 1),
        )

        self.branch1_1 = ConvBlock(dims[0] // 4, dims[0], (1, 3), 1, (0, 3 // 2))
        self.branch1_2 = ConvBlock(dims[0] // 4, dims[1], (3, 1), 1, (3 // 2, 0))
        self.branch2 = ConvBlock(in_channels, dims[2] // 4, 1, 1, 0)

        self.branch2_1 = ConvBlock(dims[2] // 4, dims[2], (1, 3), 1, (0, 3 // 2))

        self.branch2_2 = ConvBlock(dims[2] // 4, dims[3], (3, 1), 1, (3 // 2, 0))

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, dims[4], 1, 1, 0),
        )

        self.branch4 = ConvBlock(in_channels, dims[5], 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        b1 = self.branch1(x)
        b1_1 = self.branch1_1(b1)
        b1_2 = self.branch1_2(b1)
        b2 = self.branch2(x)
        b2_1 = self.branch2_1(b2)
        b2_2 = self.branch2_2(b2)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat(
            [b1_1, b1_2, b2_1, b2_2, b3, b4],
            dim=1,
        )


class GridReduction(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(inp, outp, 3, 2, 0)
        self.pooling = nn.MaxPool2d(3, 2)

    def forward(self, x: Tensor) -> Tensor:
        conv = self.conv(x)
        pooling = self.pooling(x)
        return torch.cat([conv, pooling], dim=1)


class AuxClassifier(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
    ) -> None:
        super().__init__()
        self.pooling = nn.AvgPool2d(5, 3)
        self.conv_1 = ConvBlock(inp, 128, 1, 1, 1)
        self.conv_2 = ConvBlock(128, 768, 5, 1, 1)
        self.flatten = nn.Flatten()
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, outp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pooling(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.avgpooling(x)
        x = self.flatten(x)
        return self.fc(x)
