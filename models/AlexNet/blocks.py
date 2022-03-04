import torch
import torch.nn as nn
from typing import List

__all__ = ["ConvBlock", "Classifier"]


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


class Classifier(nn.Module):

    def __init__(
            self,
            dim: List[int],
            dropout_rate: float,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(dim[0], dim[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim[1], dim[1]),
            nn.ReLU(),
            nn.Linear(dim[1], dim[2]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)