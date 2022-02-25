import torch
import torch.nn as nn
import typing


class LeNet5(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(in_channels, 6),
            ConvBlock(6, 16),
            ConvBlock(16, 120, poolling=False),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        bias: bool = False,
        poolling: bool = True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            ),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2) if poolling else nn.Identity(),
        )

    def forward(self, x: torch.Tensor):
        x =  self.block(x)
        return x 
