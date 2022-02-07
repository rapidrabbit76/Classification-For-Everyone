import torch
from torch import nn

from typing import Tuple, Union, List, Any

__all__ = ["ConvBlock", "WideResNetBlock", "Classifier"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ):
        super().__init__()
        dropout_rate = kwargs.get("dropout_rate", 0)
        layer = []
        # WRN use BN->ReLU->Conv
        layer += [nn.BatchNorm2d(out_channels)]
        layer += [nn.ReLU(inplace=True)]
        if dropout_rate > 0:
            layer += [nn.Dropout2d(dropout_rate, inplace=True)]
        layer += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kwargs.get("kernel_size"),
                stride=kwargs.get("stride", 1),
                padding=kwargs.get("padding", 0),
                groups=kwargs.get("groups", 1),
                bias=kwargs.get("bias", False),
            )
        ]

        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WideResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_rate: int,
        **kwargs: Any,
    ):
        super().__init__()
        # groups = kwargs.get("groups", 1)
        # depth = kwargs.get("depth", 64)
        # basewidth = kwargs.get("basewidth", 64)
        # width = int(depth * out_channels / basewidth) * groups

        self.residual = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                **kwargs,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dropout_rate=dropout_rate,
                **kwargs,
            ),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        sc = self.shortcut(x)
        return residual + sc


class Classifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)
