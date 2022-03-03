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
        self.conv1 = ConvBlock(image_channels, 32, 3, 2)
        self.conv2 = ConvBlock(32, 64, 3)

        # Entry Flow Xception Blocks
        self.xception_block1 = XceptionBlock(
            inp=64,
            outp=128,
            reps=2,
            s=2,
            start_act=False,
            grow_first=True,
        )
        self.xception_block2 = XceptionBlock(
            inp=128,
            outp=256,
            reps=2,
            s=2,
            start_act=True,
            grow_first=True,
        )
        self.xception_block3 = XceptionBlock(
            inp=256,
            outp=728,
            reps=2,
            s=2,
            start_act=True,
            grow_first=True,
        )

        self.middle_flow_xception_blocks = nn.Sequential(
            *[
                XceptionBlock(
                    inp=728,
                    outp=728,
                    reps=3,
                    s=1,
                    start_act=True,
                    grow_first=True,
                )
                for _ in range(8)
            ]
        )

        self.xception_block4 = XceptionBlock(
            inp=728,
            outp=1024,
            reps=2,
            s=2,
            start_act=True,
            grow_first=False,
        )

        self.conv3 = SeparableConv(1024, 1536)
        self.conv4 = SeparableConv(1536, 2048)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry flow
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.xception_block1(x)
        x = self.xception_block2(x)
        x = self.xception_block3(x)

        # Middle flow
        x = self.middle_flow_xception_blocks(x)

        # Exit flow
        x = self.xception_block4(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Classification Flow
        x = self.pooling(x)
        x = self.classifier(x)
        return x
