import torch
from torch import nn

from typing import Tuple, Union, List, Any

__all__ = ["ConvBlock", "WideResNetBlock", "Classifier"]


class ConvBlock(nn.Module):
    def __init__(
        self, inp: int, outp: int, k: int, s: int = 1, p: int = 0, dr: int = 0
    ):
        super().__init__()
        # WRN use BN->ReLU->Conv
        layer = [nn.BatchNorm2d(inp)]
        layer += [nn.ReLU(inplace=True)]
        if dr > 0:
            layer += [nn.Dropout2d(dr)]
        layer += [nn.Conv2d(inp, outp, k, s, p, bias=False)]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WideResNetBlock(nn.Module):
    def __init__(self, inp: int, outp: int, s: int, dr: int):
        super().__init__()

        self.residual = nn.Sequential(
            ConvBlock(inp, outp, 3, s, 1),
            ConvBlock(outp, outp, 3, 1, 1, dr=dr),
        )

        self.shortcut = nn.Sequential()
        if s != 1 or inp != outp:
            self.shortcut = nn.Conv2d(inp, outp, 1, s, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        sc = self.shortcut(x)
        return residual + sc


class Classifier(nn.Module):
    def __init__(self, inp: int, outp: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            in_features=inp,
            out_features=outp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)
