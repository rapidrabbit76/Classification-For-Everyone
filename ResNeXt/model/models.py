import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *

__all__ = ['ResNext', 'model_types']

model_types = {
    'resnext50': [3, 4, 6, 3],
    'resnext101': [3, 4, 23, 3],
    'resnext152': [3, 4, 36, 3],
}


class ResNext(nn.Module):

    def __init__(
        self,
        model_type: str,
        image_channels: int,
        num_classes: int,
        norm: int,
        groups: int,
        depth: int,
        basewidth: int,
    ):
        super().__init__()
        self.in_channels = 64

        num_blocks = model_types[model_type]

        self.conv1 = ConvBlock(
            in_channels=image_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            norm=norm,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        kwargs = {
            'groups': groups,
            'depth': depth,
            'basewidth': basewidth,
            'norm': norm,
        }
        self.conv2 = self._make_layer(num_blocks[0], 64, 1, **kwargs)
        self.conv3 = self._make_layer(num_blocks[1], 128, 2, **kwargs)
        self.conv4 = self._make_layer(num_blocks[2], 256, 2, **kwargs)
        self.conv5 = self._make_layer(num_blocks[3], 512, 2, **kwargs)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(
            in_features=512 * 4,
            num_classes=num_classes,
        )

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
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    **kwargs,
                )
            ]
            self.in_channels = out_channels * 4

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
