import torch
from torch import nn
from typing import Tuple, Union, List, Any

__all__ = [
    "ConvBlock",
    "ShuffleNetUnit",
    "Classifier",
]


Normalization = nn.BatchNorm2d
Activation = nn.ReLU


class ConvBlock(nn.Module):
    def __init__(
        self, inp: int, outp: int, k: int, s: int = 1, p: int = 0, act: bool = True
    ):
        super().__init__()
        layer = [nn.Conv2d(inp, outp, k, s, p, bias=False)]
        layer += [Normalization(outp)]
        if act:
            layer += [Activation(inplace=True)]

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
    def __init__(self, inp: int, k: int, s: int = 1, p: int = 0) -> None:
        super().__init__()
        outp = inp
        layer = [nn.Conv2d(inp, outp, k, s, p, groups=inp, bias=False)]
        layer += [Normalization(outp)]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ShuffleNetUnit(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        s: int,
    ) -> None:
        super().__init__()
        self.sc = nn.Identity()

        outp = outp // 2
        if s > 1:
            self.sc = nn.Sequential(
                DepthWiseConvBlock(inp, 3, s, 1),
                ConvBlock(inp, outp, 1),
            )

        self.conv = nn.Sequential(
            ConvBlock(inp=inp if s > 1 else outp, outp=outp, k=1),
            DepthWiseConvBlock(outp, 3, s, 1),
            ConvBlock(outp, outp, 1),
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
        inp: int,
        outp: int,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(inp, outp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)
