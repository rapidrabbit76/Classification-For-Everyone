import torch
import torch.nn as nn
from .blocks import *

__all__ = [
    "AlexNet"
]


class AlexNet(nn.Module):

    def __init__(
            self,
            image_channels: int,
            num_classes: int,
            dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(
                in_channels=image_channels,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=2,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBlock(
                in_channels=64,
                out_channels=192,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBlock(
                in_channels=192,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBlock(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBlock(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(6),
        )

        self.classifier = Classifier(
            dim=[256*6*6, 4096, num_classes],
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits