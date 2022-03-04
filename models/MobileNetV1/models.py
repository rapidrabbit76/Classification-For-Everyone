import torch
import torch.nn as nn
from .blocks import *


class MobileNetV1(nn.Module):

    def __init__(
            self,
            image_channels: int,
            num_classes: int,
            alpha: float = 1.0,
            dropout_rate: float = 0.5
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(
                in_channels=image_channels,
                out_channels=int(alpha*32),
                kernel_size=3,
                stride=2,
                padding=1
            ),
            DepthWiseSeparableBlock(
                dim=[int(alpha*32), int(alpha*64), int(alpha*128)],
                stride=[1, 1, 2, 1],
                is_common=True
            ),
            DepthWiseSeparableBlock(
                dim=[int(alpha*128), int(alpha*128), int(alpha*256)],
                stride=[1, 1, 2, 1],
                is_common=True
            ),
            DepthWiseSeparableBlock(
                dim=[int(alpha*256), int(alpha*256), int(alpha*512)],
                stride=[1, 1, 2, 1],
                is_common=True
            ),
            DepthWiseSeparableBlock(
                dim=[int(alpha*512)],
                stride=[1],
                iterate=5,
                is_repeat=True
            ),
            DepthWiseSeparableBlock(
                dim=[int(alpha*512), int(alpha*1024), int(alpha*1024)],
                stride=[2, 1, 2, 1],
                is_last=True
            ),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = Classifier(
            in_features=int(alpha*1024),
            out_features=num_classes,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def MobileNetV1_10(
        image_channels: int,
        num_classes: int,
        alpha: float = 1.0,
        dropout_rate: float = 0.5
):
    return MobileNetV1(image_channels, num_classes, alpha, dropout_rate)


def MobileNetV1_075(
        image_channels: int,
        num_classes: int,
        alpha: float = 0.75,
        dropout_rate: float = 0.5
):
    return MobileNetV1(image_channels, num_classes, alpha, dropout_rate)


def MobileNetV1_05(
        image_channels: int,
        num_classes: int,
        alpha: float = 0.5,
        dropout_rate: float = 0.5
):
    return MobileNetV1(image_channels, num_classes, alpha, dropout_rate)