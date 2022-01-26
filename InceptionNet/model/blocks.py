import torch
from torch import nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
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
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class Inceptionx3(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # TODO


class Inceptionx5(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # TODO


class Inceptionx2(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # TODO