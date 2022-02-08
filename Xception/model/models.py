import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *

__all__ = [
    "XceptionNet",
]


class XceptionNet(nn.Module):
    def __init__(
        self,
        image_channels: int,
        num_classes: int,
    ):
        super().__init__()

        # Entry Flow
        self.conv1 = ConvBlock(
            in_channels=image_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
        )
        self.conv2 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
        )

        # Entry Flow Xception Blocks
        self.xeption_block1 = XceptionBlock(
            in_channels=64,
            out_channels=128,
            reps=2,
            stride=2,
            start_with_relu=False,
            grow_first=True,
        )
        self.xeption_block2 = XceptionBlock(
            in_channels=128,
            out_channels=256,
            reps=2,
            stride=2,
            start_with_relu=True,
            grow_first=True,
        )
        self.xeption_block3 = XceptionBlock(
            in_channels=256,
            out_channels=728,
            reps=2,
            stride=2,
            start_with_relu=True,
            grow_first=True,
        )

        self.middle_flow_xeption_blocks = [
            XceptionBlock(
                in_channels=728,
                out_channels=728,
                reps=3,
                stride=1,
                start_with_relu=True,
                grow_first=True,
            )
            for _ in range(8)
        ]

        self.xeption_block4 = XceptionBlock(
            in_channels=728,
            out_channels=1024,
            reps=2,
            stride=2,
            start_with_relu=True,
            grow_first=False,
        )

        self.conv3 = SeparableConv(
            in_channels=1024,
            out_channels=1536,
        )
        self.conv4 = SeparableConv(
            in_channels=1536,
            out_channels=2048,
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(
            in_features=2048,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry flow
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.xeption_block1(x)
        x = self.xeption_block2(x)
        x = self.xeption_block2(x)

        # Middle flow
        x = self.middle_flow_xeption_blocks(x)

        # Exit flow
        x = self.xeption_block4(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Claaification Flow
        x = self.pooling(x)
        x = self.classifier(x)
        return x

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
