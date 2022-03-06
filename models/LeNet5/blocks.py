import torch
import torch.nn as nn
from typing import *


class ConvBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        k: int = 5,
        s: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(inp, outp, k, s, bias=False)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.act(x)
