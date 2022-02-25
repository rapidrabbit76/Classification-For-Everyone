import torch
from torch import nn

__all__ = ["ConvBlock", "Classifier"]

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


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
        self.norm = Normalization(out_channels)
        self.act = Activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class Classifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        dim: int,
        out_features: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
