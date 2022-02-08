import torch
from torch import nn

from typing import Tuple, Union, List, Any

__all__ = [
    "ConvBlock",
    "XceptionBlock",
    "SeparableConv",
    "Classifier",
]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ):
        super().__init__()
        layer = []
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
        layer += [nn.BatchNorm2d(out_channels)]
        if kwargs.get("act", True):
            layer += [nn.ReLU(inplace=True)]

        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=kwargs.get("bias", False),
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=kwargs.get("bias", False),
        )
        self.norm = nn.BatchNorm2d(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.norm(x)


class XceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reps: int,
        stride: int,
        start_with_relu: bool,
        grow_first: bool,
    ):
        super(XceptionBlock, self).__init__()

        self.skip = nn.Identity()
        if out_channels != in_channels or stride != 1:
            self.skip = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                act=False,
            )

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_channels
        if grow_first:
            rep += [self.relu]
            rep += [SeparableConv(in_channels, out_channels)]
            filters = out_channels

        for _ in range(reps - 1):
            rep += [self.relu]
            rep += [SeparableConv(filters, filters)]

        if not grow_first:
            rep += [self.relu]
            rep += [SeparableConv(in_channels, out_channels)]

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rep(x) + self.skip(x)


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
