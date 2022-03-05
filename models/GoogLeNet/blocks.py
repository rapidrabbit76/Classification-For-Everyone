import torch
import torch.nn as nn

__all__ = ["ConvBlock", "InceptionBlock", "Classifier"]


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            bias: bool = False,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        ]

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class InceptionBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels_1by1: int,
            out_channels_3by3_reduce: int,
            out_channels_3by3: int,
            out_channels_5by5_reduce: int,
            out_channels_5by5: int,
            out_channels_pool_proj: int,
    ):
        super().__init__()
        self.branch1 = ConvBlock(in_channels, out_channels_1by1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, out_channels_3by3_reduce, kernel_size=1),
            ConvBlock(out_channels_3by3_reduce, out_channels_3by3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, out_channels_5by5_reduce, kernel_size=1),
            ConvBlock(out_channels_5by5_reduce, out_channels_5by5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_channels_pool_proj, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat([x1, x2, x3, x4], dim=1)


class Classifier(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout_rate: float,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)