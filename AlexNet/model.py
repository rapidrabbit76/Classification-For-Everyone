import torch
import torch.nn as nn
import typing


class AlexNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            n_classes: int,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(in_channels, 96, kernel_size=11, stride=4),
            ConvBlock(96, 256, kernel_size=5, padding='same'),
            ConvBlock(256, 384, kernel_size=3, padding='same', pooling=False, use_lrm=False),
            ConvBlock(384, 384, kernel_size=3, padding='same', pooling=False, use_lrm=False),
            ConvBlock(384, 256, kernel_size=3, padding='same')
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256*6*6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=n_classes),
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
            kernel_size: int,
            stride: int = 1,
            padding: str = 'valid',
            bias: bool = True,
            pooling: bool = True,
            use_lrm: bool = True,
            use_bn: bool = False,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2) if use_lrm else nn.Identity(),
            nn.BatchNorm2d(num_features=out_channels) if not use_lrm and use_bn else nn.Identity(),
            nn.MaxPool2d(kernel_size=3, stride=2) if pooling else nn.Identity(),
        )

    def forward(self, x: torch.Tensor):
        x = self.block(x)
        return x