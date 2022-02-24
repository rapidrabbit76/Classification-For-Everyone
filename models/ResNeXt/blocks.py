import torch
from torch import nn

from typing import *

__all__ = ["ConvBlock", "ResNeXtBlock", "Classifier"]

Activation = nn.ReLU


class ConvBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        k: int,
        s: int = 1,
        p: int = 0,
        g: int = 1,
        act: bool = True,
    ):
        super().__init__()
        layer = [nn.Conv2d(inp, outp, k, s, p, groups=g, bias=False)]
        layer += [nn.BatchNorm2d(outp)]
        if act:
            layer += [nn.ReLU(inplace=True)]

        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNeXtBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        s: int,
        g: int,
        depth: int,
        basewidth: int,
    ):
        super().__init__()

        width = int(depth * outp / basewidth) * g
        self.split_transforms = nn.Sequential(
            ConvBlock(inp, width, 1),
            ConvBlock(width, width, 3, s, 1, g),
            ConvBlock(width, outp * 4, 1, act=False),
        )

        self.shortcut = nn.Sequential()
        self.act = nn.ReLU(inplace=True)

        if s != 1 or inp != outp * 4:
            self.shortcut = ConvBlock(inp, outp * 4, 1, s, act=False)

    def forward(self, x):
        residual = self.split_transforms(x)
        sc = self.shortcut(x)
        return self.act(residual + sc)


class Classifier(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            in_features=inp,
            out_features=outp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)
