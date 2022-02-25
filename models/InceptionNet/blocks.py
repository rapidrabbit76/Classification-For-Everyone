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
        padding: _size_2_t = 1,
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
        self.norm = nn.BatchNorm2d(out_channels,
                                   eps=0.001) if norm else nn.Identity()
        self.act = nn.ReLU()

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
            ConvBlock(in_channels, dims[2], 1, stride=1, padding=0, norm=norm),
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

    def __init__(
        self,
        in_channels: int,
        dims: List[int],
        norm: bool,
        n: int = 7,
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(
                in_channels,
                dims[0],
                1,
                stride=1,
                padding=0,
                norm=norm,
            ),
            ConvBlock(
                dims[0],
                dims[0],
                kernel_size=(1, n),
                stride=1,
                padding=(0, n // 2),
                norm=norm,
            ),
            ConvBlock(
                dims[0],
                dims[0],
                kernel_size=(n, 1),
                stride=1,
                padding=(n // 2, 0),
                norm=norm,
            ),
            ConvBlock(
                dims[0],
                dims[0],
                kernel_size=(1, n),
                stride=1,
                padding=(0, n // 2),
                norm=norm,
            ),
            ConvBlock(
                dims[0],
                dims[0],
                kernel_size=(n, 1),
                stride=1,
                padding=(n // 2, 0),
                norm=norm,
            ),
        )

        self.branch2 = nn.Sequential(
            ConvBlock(
                in_channels,
                dims[1],
                kernel_size=1,
                stride=1,
                padding=0,
                norm=norm,
            ),
            ConvBlock(
                dims[1],
                dims[1],
                kernel_size=(1, n),
                stride=1,
                padding=(0, n // 2),
                norm=norm,
            ),
            ConvBlock(
                dims[1],
                dims[1],
                kernel_size=(n, 1),
                stride=1,
                padding=(n // 2, 0),
                norm=norm,
            ),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            ConvBlock(
                in_channels,
                dims[2],
                kernel_size=1,
                stride=1,
                padding=0,
                norm=norm,
            ),
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


class Inceptionx2(nn.Module):

    def __init__(
        self,
        in_channels: int,
        dims: List[int],
        norm: bool,
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(
                in_channels,
                dims[0] // 4,
                kernel_size=1,
                stride=1,
                padding=0,
                norm=norm,
            ),
            ConvBlock(
                dims[0] // 4,
                dims[0] // 4,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=norm,
            ),
        )

        self.branch1_1 = ConvBlock(
            dims[0] // 4,
            dims[0],
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 3 // 2),
            norm=norm,
        )

        self.branch1_2 = ConvBlock(
            dims[0] // 4,
            dims[1],
            kernel_size=(3, 1),
            stride=1,
            padding=(3 // 2, 0),
            norm=norm,
        )

        self.branch2 = ConvBlock(
            in_channels,
            dims[2] // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=norm,
        )

        self.branch2_1 = ConvBlock(
            dims[2] // 4,
            dims[2],
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 3 // 2),
            norm=norm,
        )

        self.branch2_2 = ConvBlock(
            dims[2] // 4,
            dims[3],
            kernel_size=(3, 1),
            stride=1,
            padding=(3 // 2, 0),
            norm=norm,
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(
                in_channels,
                dims[4],
                kernel_size=1,
                stride=1,
                padding=0,
                norm=norm,
            ))

        self.branch4 = ConvBlock(
            in_channels,
            dims[5],
            kernel_size=1,
            stride=1,
            padding=0,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b1_1 = self.branch1_1(b1)
        b1_2 = self.branch1_2(b1)
        b2 = self.branch2(x)
        b2_1 = self.branch2_1(b2)
        b2_2 = self.branch2_2(b2)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat(
            [
                b1_1,
                b1_2,
                b2_1,
                b2_2,
                b3,
                b4,
            ],
            dim=1,
        )


class GridReduction(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            norm=norm,
        )

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv(x)
        pooling = self.pooling(x)
        return torch.cat([conv, pooling], dim=1)


class AuxClassifier(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        norm: bool,
    ) -> None:
        super().__init__()
        self.pooling = nn.AvgPool2d(
            kernel_size=(5, 5),
            stride=3,
        )
        self.conv_1 = ConvBlock(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=1,
            norm=norm,
        )
        self.conv_2 = ConvBlock(
            in_channels=128,
            out_channels=768,
            kernel_size=5,
            stride=1,
            padding=1,
            norm=norm,
        )
        self.flatten = nn.Flatten()
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pooling(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.avgpooling(x)
        x = self.flatten(x)
        return self.fc(x)
