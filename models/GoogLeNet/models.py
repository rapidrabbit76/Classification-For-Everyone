import torch
import torch.nn as nn
from .blocks import *

__all__ = ["GoogLeNet"]


class GoogLeNet(nn.Module):
    def __init__(
        self,
        image_channels: int,
        num_classes: int,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(image_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3),
            ConvBlock(64, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=192, eps=1e-3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # inception (3a)
            InceptionBlock(
                in_channels=192,
                out_channels_1by1=64,
                out_channels_3by3_reduce=96,
                out_channels_3by3=128,
                out_channels_5by5_reduce=16,
                out_channels_5by5=32,
                out_channels_pool_proj=32,
            ),
            # inception (3b)
            InceptionBlock(
                in_channels=256,
                out_channels_1by1=128,
                out_channels_3by3_reduce=128,
                out_channels_3by3=192,
                out_channels_5by5_reduce=32,
                out_channels_5by5=96,
                out_channels_pool_proj=64,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # inception (4a)
            InceptionBlock(
                in_channels=480,
                out_channels_1by1=192,
                out_channels_3by3_reduce=96,
                out_channels_3by3=208,
                out_channels_5by5_reduce=16,
                out_channels_5by5=48,
                out_channels_pool_proj=64,
            ),
            # inception (4b)
            InceptionBlock(
                in_channels=512,
                out_channels_1by1=160,
                out_channels_3by3_reduce=112,
                out_channels_3by3=224,
                out_channels_5by5_reduce=24,
                out_channels_5by5=64,
                out_channels_pool_proj=64,
            ),
            # inception (4c)
            InceptionBlock(
                in_channels=512,
                out_channels_1by1=128,
                out_channels_3by3_reduce=128,
                out_channels_3by3=256,
                out_channels_5by5_reduce=24,
                out_channels_5by5=64,
                out_channels_pool_proj=64,
            ),
            # inception (4d)
            InceptionBlock(
                in_channels=512,
                out_channels_1by1=112,
                out_channels_3by3_reduce=144,
                out_channels_3by3=288,
                out_channels_5by5_reduce=32,
                out_channels_5by5=64,
                out_channels_pool_proj=64,
            ),
            # inception (4e)
            InceptionBlock(
                in_channels=528,
                out_channels_1by1=256,
                out_channels_3by3_reduce=160,
                out_channels_3by3=320,
                out_channels_5by5_reduce=32,
                out_channels_5by5=128,
                out_channels_pool_proj=128,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # inception (5a)
            InceptionBlock(
                in_channels=832,
                out_channels_1by1=256,
                out_channels_3by3_reduce=160,
                out_channels_3by3=320,
                out_channels_5by5_reduce=32,
                out_channels_5by5=128,
                out_channels_pool_proj=128,
            ),
            # inception (5b)
            InceptionBlock(
                in_channels=832,
                out_channels_1by1=384,
                out_channels_3by3_reduce=192,
                out_channels_3by3=384,
                out_channels_5by5_reduce=48,
                out_channels_5by5=128,
                out_channels_pool_proj=128,
            ),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = Classifier(
            in_features=1024, out_features=num_classes, dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
