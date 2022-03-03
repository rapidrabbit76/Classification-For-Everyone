from typing import *

import torch
from torch import nn

__all__ = [
    "ConvBlock",
    "XceptionBlock",
    "SeparableConv",
    "Classifier",
]


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


class SeparableConv(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        **kwargs,
    ):
        super().__init__()
        layer = [nn.Conv2d(inp, inp, 3, padding=1, groups=inp, bias=False)]
        layer += [nn.Conv2d(inp, outp, 1, bias=False)]
        layer += [nn.BatchNorm2d(outp)]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class XceptionBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        reps: int,
        s: int,
        start_act: bool,
        grow_first: bool,
    ):
        super().__init__()

        self.skip = nn.Identity()
        if outp != inp or s != 1:
            self.skip = ConvBlock(inp, outp, 1, s, act=False)

        act = nn.ReLU(inplace=True)

        layer = []
        filters = inp
        if grow_first:
            layer += [act]
            layer += [SeparableConv(inp, outp)]
            filters = outp

        for _ in range(reps - 1):
            layer += [act]
            layer += [SeparableConv(filters, filters)]

        if not grow_first:
            layer += [act]
            layer += [SeparableConv(inp, outp)]

        if not start_act:
            layer = layer[1:]
        else:
            layer[0] = nn.ReLU(inplace=False)

        if s != 1:
            layer += [nn.MaxPool2d(3, s, 1)]

        self.rep = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rep(x) + self.skip(x)


class Classifier(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(inp, outp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)
