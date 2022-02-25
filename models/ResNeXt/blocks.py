from asyncio import FastChildWatcher
from numpy import number
import torch
from torch import nn

from typing import Tuple, Union, List, Any

__all__ = ['ConvBlock', 'ResNeXtBlock', 'Classifier']


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ):
        super().__init__()
        act = kwargs.get('act', True)
        layer = []

        layer += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kwargs.get('kernel_size'),
                stride=kwargs.get('stride', 1),
                padding=kwargs.get('padding', 0),
                groups=kwargs.get('groups', 1),
                bias=kwargs.get('bias', False),
            )
        ]
        layer += [nn.BatchNorm2d(out_channels)]
        if act:
            layer += [nn.ReLU(inplace=True)]

        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNeXtBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        **kwargs: Any,
    ):
        super().__init__()
        groups = kwargs.get('groups')
        depth = kwargs.get('depth')
        basewidth = kwargs.get('basewidth')

        width = int(depth * out_channels / basewidth) * groups
        self.split_transforms = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=width,
                kernel_size=1,
            ),
            ConvBlock(
                in_channels=width,
                out_channels=width,
                kernel_size=3,
                stride=stride,
                groups=groups,
                padding=1,
            ),
            ConvBlock(
                in_channels=width,
                out_channels=out_channels * 4,
                kernel_size=1,
                act=False,
            ),
        )

        self.shortcut = nn.Sequential()
        self.act = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                act=False,
            )

    def forward(self, x):
        residual = self.split_transforms(x)
        sc = self.shortcut(x)

        return self.act(residual + sc)


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
