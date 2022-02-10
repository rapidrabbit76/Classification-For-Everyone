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


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int) -> None:
        super().__init__()
        self.__groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        channels_per_group = c // self.__groups
        x = x.view(b, self.__groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(b, -1, h, w)


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
