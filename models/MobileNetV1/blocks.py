import torch
import torch.nn as nn
from typing import List

__all__ = ["ConvBlock", "DepthWiseSeparableBlock", "Classifier"]


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

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        ]

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class DepthWiseSeparableBlock(nn.Module):

    def __init__(
            self,
            dim: List[int],
            stride: List[int],
            iterate: int = 0,
            is_common: bool = False,
            is_repeat: bool = False,
            is_last: bool = False,
    ) -> None:
        super().__init__()

        layers = []

        if is_common is True:
            layers.append(ConvBlock(
                in_channels=dim[0],
                out_channels=dim[0],
                kernel_size=3,
                stride=stride[0],
                padding=1,
                groups=dim[0]
            )),
            layers.append(ConvBlock(
                in_channels=dim[0],
                out_channels=dim[1],
                kernel_size=1,
                stride=stride[1]
            )),
            layers.append(ConvBlock(
                in_channels=dim[1],
                out_channels=dim[1],
                kernel_size=3,
                stride=stride[2],
                padding=1,
                groups=dim[1]
            )),
            layers.append(ConvBlock(
                in_channels=dim[1],
                out_channels=dim[2],
                kernel_size=1,
                stride=stride[3]
            ))

        elif is_repeat is True:
            for i in range(iterate):
                layers.append(ConvBlock(
                    in_channels=dim[0],
                    out_channels=dim[0],
                    kernel_size=3,
                    stride=stride[0],
                    padding=1,
                    groups=dim[0]
                )),
                layers.append(ConvBlock(
                    in_channels=dim[0],
                    out_channels=dim[0],
                    kernel_size=1,
                    stride=stride[0],
                ))

        elif is_last is True:
            layers.append(ConvBlock(
                in_channels=dim[0],
                out_channels=dim[0],
                kernel_size=3,
                stride=stride[0],
                padding=1,
                groups=dim[0]
            )),
            layers.append(ConvBlock(
                in_channels=dim[0],
                out_channels=dim[1],
                kernel_size=1,
                stride=stride[1]
            )),
            layers.append(ConvBlock(
                in_channels=dim[1],
                out_channels=dim[1],
                kernel_size=3,
                stride=stride[2],
                padding=4,
                groups=dim[1]
            )),
            layers.append(ConvBlock(
                in_channels=dim[1],
                out_channels=dim[2],
                kernel_size=1,
                stride=stride[3]
            ))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


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