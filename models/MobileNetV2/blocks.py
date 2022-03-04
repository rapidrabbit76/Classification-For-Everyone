import torch
import torch.nn as nn
from typing import List

__all__ = ["ConvBlock", "BottleNeckS1Block", "BottleNeckS2Block", "BottleNeck", "Classifier"]


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            act: str = 'ReLU6',
            bias: bool = False,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(num_features=out_channels)
        ]

        if act == 'ReLU6':
            layers.append(nn.ReLU6())

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class BottleNeckS1Block(nn.Module):

    def __init__(
            self,
            dim: List[int],
            factor: int,
    ) -> None:
        super().__init__()
        self.use_res_connection = dim[0] == dim[1]

        self.blocks = nn.Sequential(
            # Conv 1x1, ReLU6
            ConvBlock(
                in_channels=dim[0],
                out_channels=expand_width(dim[0], factor),
                kernel_size=1,
                act='ReLU6',
            ),
            # Dwise 3x3, ReLU6
            ConvBlock(
                in_channels=expand_width(dim[0], factor),
                out_channels=expand_width(dim[0], factor),
                kernel_size=3,
                padding=1,
                groups=expand_width(dim[0], factor),
                act='ReLU6',
            ),
            # Conv 1x1, linear act
            ConvBlock(
                in_channels=expand_width(dim[0], factor),
                out_channels=dim[1],
                kernel_size=1,
                act='None',
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connection is True:
            identity = x
            return identity + self.blocks(x)
        else:
            return self.blocks(x)


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
                out_channels=expand_width(dim[0], factor),
                kernel_size=1,
                act='ReLU6',
            ),
            # Dwise 3x3, ReLU6
            ConvBlock(
                in_channels=expand_width(dim[0], factor),
                out_channels=expand_width(dim[0], factor),
                kernel_size=3,
                stride=2,
                padding=1,
                groups=expand_width(dim[0], factor),
                act='ReLU6',
            ),
            # Conv 1x1, linear act
            ConvBlock(
                in_channels=expand_width(dim[0], factor),
                out_channels=dim[1],
                kernel_size=1,
                act='None',
            ),
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


def expand_width(
        dim: int,
        factor: float,
) -> int:
    return int(dim*factor)


class Classifier(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout_rate: float,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)