from asyncio import FastChildWatcher
import torch
from torch import nn

from typing import Tuple, Union, List, Any

__all__ = ['ConvBlock', 'ResNeXtBlock']


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        act: bool,
        **kwargs: Any,
    ):
        super().__init__()
        kwargs['bias'] = False
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResNeXtBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm: bool,
        groups: int,
        depth: int,
        base_width: int,
        **kwargs,
    ) -> None:
        super().__init__()

        d = int(depth * out_channels / base_width)

        self.residual_block = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=groups * d,
                norm=norm,
                act=True,
                kernel_size=1,
                stride=1,
                padding=0,
                kwargs=kwargs,
            ),
            ConvBlock(
                in_channels=groups * d,
                out_channels=groups * d,
                norm=norm,
                act=True,
                kernel_size=3,
                stride=stride,
                padding=1,
                kwargs=kwargs,
            ),
            ConvBlock(
                in_channels=groups * d,
                out_channels=out_channels * 4,
                norm=norm,
                act=False,
                kernel_size=1,
                stride=1,
                kwargs=kwargs,
            ),
        )
        self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                norm=norm,
                act=False,
                stride=stride,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual_block(x)
        sc = self.shortcut(x)
        return self.act(res + sc)
