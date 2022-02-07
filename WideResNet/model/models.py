import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *

__all__ = ["WideResNet", "model_types"]


class WideResNet(nn.Module):
    def __init__(
        self,
        image_channels: int,
        num_classes: int,
        depth: int,
        K: int,
        dropout_rate: int,
    ):
        super().__init__()
        N = (depth - 4) // 6
        self.in_channels = 16

        self.conv1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        kwargs = {}
        self.conv2 = self._make_layer(16 * K, N, 1, dropout_rate, **kwargs)
        self.conv3 = self._make_layer(32 * K, N, 2, dropout_rate, **kwargs)
        self.conv4 = self._make_layer(64 * K, N, 2, dropout_rate, **kwargs)
        self.bn = nn.BatchNorm2d(64 * K)
        self.act = nn.ReLU(inplace=True)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(
            in_features=64 * K,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.avg(x)
        return self.classifier(x)

    def _make_layer(
        self,
        out_channels: int,
        num_block: int,
        stride: int,
        dropout_rate: float,
        **kwargs: any,
    ):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers += [
                WideResNetBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dropout_rate=dropout_rate,
                    **kwargs,
                )
            ]
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
