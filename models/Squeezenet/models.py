from math import log
from xml.etree.ElementInclude import include
import torch
from torch import nn
import math

from .blocks import FireModule, ConvBlock


class SqueezeNetFeatureExtractor(nn.Module):

    def __init__(self, image_channals: int) -> None:
        super().__init__()

        self.conv_1 = ConvBlock(image_channals, 96, 7, 2)
        self.maxpool_1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.fire_2 = FireModule(96, 16, 64, 64)
        self.fire_3 = FireModule(128, 16, 64, 64)
        self.fire_4 = FireModule(128, 32, 128, 128)
        self.maxpool_4 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.fire_5 = FireModule(256, 32, 128, 128)
        self.fire_6 = FireModule(256, 48, 192, 192)
        self.fire_7 = FireModule(384, 48, 192, 192)
        self.fire_8 = FireModule(384, 64, 256, 256)
        self.maxpool_8 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.fire_9 = FireModule(512, 64, 256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.fire_2(x)
        x = self.fire_3(x)
        x = self.fire_4(x)
        x = self.maxpool_4(x)
        x = self.fire_5(x)
        x = self.fire_6(x)
        x = self.fire_7(x)
        x = self.fire_8(x)
        x = self.maxpool_8(x)
        return self.fire_9(x)


class SqueezeNet(nn.Module):

    def __init__(
        self,
        image_channals: int = 3,
        n_classes: int = 1000,
    ) -> None:
        super().__init__()

        self.feature_extractor = SqueezeNetFeatureExtractor(image_channals)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            ConvBlock(512, n_classes, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return self.classifier(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight,
                    nonlinearity='relu',
                )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
