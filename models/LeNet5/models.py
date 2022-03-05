import torch
import torch.nn as nn
import typing
from .blocks import ConvBlock


class LeNet5(nn.Module):
    def __init__(
        self,
        image_channels: int,
        num_classes: int,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(image_channels, 6),
            nn.AvgPool2d(2),
            ConvBlock(6, 16),
            nn.AvgPool2d(2),
            ConvBlock(16, 120),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
