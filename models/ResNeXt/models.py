import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *

__all__ = [
    "ResNeXt50",
    "ResNeXt101",
    "ResNeXt152",
]

model_types = {
    "50": [3, 4, 6, 3],
    "101": [3, 4, 23, 3],
    "152": [3, 4, 36, 3],
}


class ResNext(nn.Module):
    def __init__(
        self,
        model_type: str,
        image_channels: int,
        num_classes: int,
        groups: int = 32,
        depth: int = 4,
        basewidth: int = 64,
    ):
        super().__init__()
        self.in_channels = 64

        num_blocks = model_types[model_type]

        self.conv1 = ConvBlock(image_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        kwargs = {
            "g": groups,
            "depth": depth,
            "basewidth": basewidth,
        }
        self.conv2 = self._make_layer(num_blocks[0], 64, 1, **kwargs)
        self.conv3 = self._make_layer(num_blocks[1], 128, 2, **kwargs)
        self.conv4 = self._make_layer(num_blocks[2], 256, 2, **kwargs)
        self.conv5 = self._make_layer(num_blocks[3], 512, 2, **kwargs)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier(inp=512 * 4, outp=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        return self.classifier(x)

    def _make_layer(
        self,
        num_block: int,
        out_channels: int,
        stride: int,
        **kwargs: any,
    ):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers += [
                ResNeXtBlock(
                    inp=self.in_channels,
                    outp=out_channels,
                    s=stride,
                    **kwargs,
                )
            ]
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)


def ResNeXt50(
    image_channels: int,
    num_classes: int,
    groups: int = 32,
    depth: int = 4,
    basewidth: int = 64,
):
    return ResNext("50", image_channels, num_classes, groups, depth, basewidth)


def ResNeXt101(
    image_channels: int,
    num_classes: int,
    groups: int = 32,
    depth: int = 4,
    basewidth: int = 64,
):
    return ResNext("101", image_channels, num_classes, groups, depth, basewidth)


def ResNeXt152(
    image_channels: int,
    num_classes: int,
    groups: int = 32,
    depth: int = 4,
    basewidth: int = 64,
):
    return ResNext("152", image_channels, num_classes, groups, depth, basewidth)
