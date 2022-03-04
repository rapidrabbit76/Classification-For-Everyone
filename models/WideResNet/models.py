import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
import torch

__all__ = [
    "WideResNet",
]

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


class WideResNet(nn.Module):
    def __init__(
        self,
        image_channels: int,
        num_classes: int,
        depth: int = 40,
        K: int = 10,
        dropout_rate: int = 0.5,
    ):
        super().__init__()
        N = (depth - 4) // 6
        self.in_channels = 16

        self.conv1 = nn.Conv2d(image_channels, 16, 3, 1, 1)
        self.conv2 = self._make_layer(16 * K, N, 1, dropout_rate)
        self.conv3 = self._make_layer(32 * K, N, 2, dropout_rate)
        self.conv4 = self._make_layer(64 * K, N, 2, dropout_rate)
        self.bn = Normalization(self.in_channels)
        self.act = Activation(inplace=True)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier(inp=self.in_channels, outp=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.avg(x)
        return self.classifier(x)

    def _make_layer(self, out_channels: int, num_block: int, s: int, dr: float):
        layers = [WideResNetBlock(self.in_channels, out_channels, s, dr)]
        self.in_channels = out_channels

        layers += [
            WideResNetBlock(self.in_channels, out_channels, 1, dr)
            for _ in range(1, num_block)
        ]

        return nn.Sequential(*layers)
