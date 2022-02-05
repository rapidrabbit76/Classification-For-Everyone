import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BottleNeckS1Block(nn.Module):

    def __init__(
            self,
            dim: List[int],
            factor: int,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            # Conv 1x1, ReLU6
            ConvBlock(
                in_channels=dim[0],
                out_channels=dim[0]*factor,
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=dim[0]*factor),
            nn.ReLU6(),
            # Dwise 3x3, ReLU6
            ConvBlock(
                in_channels=dim[0]*factor,
                out_channels=dim[0]*factor,
                kernel_size=3,
                padding=1,
                groups=dim[0]*factor,
            ),
            nn.BatchNorm2d(num_features=dim[0]*factor),
            nn.ReLU6(),
            # Conv 1x1, linear act
            ConvBlock(
                in_channels=dim[0]*factor,
                out_channels=dim[1],
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=dim[1]),
        )
        self.downsample = ConvBlock(
            in_channels=dim[0],
            out_channels=dim[1],
            kernel_size=1,
        )
        self.bn_ds = nn.BatchNorm2d(num_features=dim[1])
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        identity = self.downsample(identity)
        identity = self.bn_ds(identity)

        x = self.blocks(x)
        x += identity

        return self.act(x)


class BottleNeckS2Block(nn.Module):

    def __init__(
            self,
            dim: List[int],
            factor: int,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            # Conv 1x1, ReLU6
            ConvBlock(
                in_channels=dim[0],
                out_channels=dim[0]*factor,
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=dim[0]*factor),
            nn.ReLU6(),
            # Dwise 3x3, ReLU6
            ConvBlock(
                in_channels=dim[0]*factor,
                out_channels=dim[0]*factor,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=dim[0]*factor,
            ),
            nn.BatchNorm2d(num_features=dim[0]*factor),
            nn.ReLU6(),
            # Conv 1x1, linear act
            ConvBlock(
                in_channels=dim[0]*factor,
                out_channels=dim[1],
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=dim[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class BottleNeck(nn.Module):

    def __init__(
            self,
            dim: List[int],
            factor: int,
            iterate: int,
            stride: int,
    ) -> None:
        super().__init__()

        layers = []

        for i in range(iterate):
            if i == 0:
                input_dim = dim[0]
                output_dim = dim[1]
            elif i != 0:
                input_dim = dim[1]
                output_dim = dim[1]

            if stride == 1:
                layers.append(BottleNeckS1Block(dim=[input_dim, output_dim], factor=factor))
            elif stride == 2:
                layers.append(BottleNeckS2Block(dim=[input_dim, output_dim], factor=factor))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Classifier(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)