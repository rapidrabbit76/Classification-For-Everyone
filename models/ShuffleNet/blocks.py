import torch
from torch import nn
from typing import Tuple, Union, List, Any

__all__ = [
    "ConvBlock",
    "ShuffleNetUnit",
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


class DepthWiseConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        out_channels = in_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            padding=kwargs.get("padding", 0),
            bias=kwargs.get("bias", False),
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.norm(x)


class ShuffleNetUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.sc = nn.Identity()

        out_channels = out_channels // 2
        if stride > 1:
            self.sc = nn.Sequential(
                DepthWiseConvBlock(
                    in_channels=in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                ),
            )

        self.conv = nn.Sequential(
            ConvBlock(
                in_channels=in_channels if stride > 1 else out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            DepthWiseConvBlock(
                in_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )
        self.channel_shuffle = ChannelShuffle(groups=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.sc, nn.Identity):
            x1, x2 = x.chunk(2, dim=1)
            x = torch.cat([self.sc(x1), self.conv(x2)], dim=1)
        else:
            x = torch.cat([self.sc(x), self.conv(x)], dim=1)
        return self.channel_shuffle(x)


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
