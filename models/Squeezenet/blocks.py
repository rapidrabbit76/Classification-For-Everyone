import torch
from torch import nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.act(x)


class FireModule(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        e1x1_channels: int,
        e3x3_channels: int,
    ):
        super().__init__()
        self.squeeze = ConvBlock(in_channels, out_channels, 1)
        self.e1x1 = ConvBlock(out_channels, e1x1_channels, 1)
        self.e3x3 = ConvBlock(
            out_channels,
            e3x3_channels,
            3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        return torch.cat([self.e1x1(x), self.e3x3(x)], 1)
