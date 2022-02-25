import torch
import torch.nn as nn
import numpy as np
from .blocks import *
from typing import Final, Dict

__all__ = [
    "EfficientNet", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6",
    "efficientnet_b7"
]

MODEL_TYPE: Final[Dict] = {
    # numLayer, input, output, expand ratio, kernel size, padding, stride, reduction ratio
    # fmt: off
    'MBConv1':   [1,  32,  16, 1.0, 3, 1, 1, 4],

    'MBConv6_1': [2,  16,  24, 6.0, 3, 1, 2, 4],

    'MBConv6_2': [2,  24,  40, 6.0, 5, 2, 2, 4],

    'MBConv6_3': [3,  40,  80, 6.0, 3, 1, 2, 4],

    'MBConv6_4': [3,  80, 112, 6.0, 5, 2, 1, 4],

    'MBConv6_5': [4, 112, 192, 6.0, 5, 2, 2, 4],

    'MBConv6_6': [1, 192, 320, 6.0, 3, 1, 1, 4]
}


class EfficientNet(nn.Module):

    def __init__(
            self,
            image_channels: int,
            num_classes: int,
            width_coefficient: float = 1.0,
            depth_coefficient: float = 1.0,
            dropout_rate: float = 0.5
    ) -> None:
        super().__init__()
        layers = []
        Blockkeys = MODEL_TYPE.keys()
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient

        layers.append(
            ConvBlock(
                in_channels=image_channels,
                out_channels=self.multiply_width(32),
                kernel_size=3,
                stride=2,
                padding=1,
            )
        ),

        for key in Blockkeys:
            layers.append(
                MBConvBlock(
                    numLayer=self.multiply_depth(numLayer=MODEL_TYPE[key][0]),
                    dim=[
                        self.multiply_width(MODEL_TYPE[key][1]),
                        self.multiply_width(MODEL_TYPE[key][2])
                    ],
                    factor=MODEL_TYPE[key][3],
                    kernel_size=MODEL_TYPE[key][4],
                    padding=MODEL_TYPE[key][5],
                    stride=MODEL_TYPE[key][6],
                    reduction_ratio=MODEL_TYPE[key][7]
                )
            )

        layers.append(
            ConvBlock(
                in_channels=self.multiply_width(MODEL_TYPE['MBConv6_6'][2]),
                out_channels=self.multiply_width(1280),
                kernel_size=1
            )
        )

        layers.append(nn.AdaptiveAvgPool2d(1))

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = Classifier(
            in_features=self.multiply_width(1280),
            out_features=num_classes,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def multiply_width(self, dim: int) -> int:
        return int(np.ceil(self.width_coefficient*dim))

    def multiply_depth(self, numLayer: int) -> int:
        return int(np.ceil(self.depth_coefficient*numLayer))


def efficientnet_b0(
        image_channels: int,
        n_classes: int,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        dropout_rate: float = 0.5
):
    return EfficientNet(image_channels, n_classes, width_coefficient, depth_coefficient, dropout_rate)


def efficientnet_b1(
        image_channels: int,
        n_classes: int,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.1,
        dropout_rate: float = 0.5
):
    return EfficientNet(image_channels, n_classes, width_coefficient, depth_coefficient, dropout_rate)


def efficientnet_b2(
        image_channels: int,
        n_classes: int,
        width_coefficient: float = 1.1,
        depth_coefficient: float = 1.2,
        dropout_rate: float = 0.5
):
    return EfficientNet(image_channels, n_classes, width_coefficient, depth_coefficient, dropout_rate)


def efficientnet_b3(
        image_channels: int,
        n_classes: int,
        width_coefficient: float = 1.2,
        depth_coefficient: float = 1.4,
        dropout_rate: float = 0.5
):
    return EfficientNet(image_channels, n_classes, width_coefficient, depth_coefficient, dropout_rate)


def efficientnet_b4(
        image_channels: int,
        n_classes: int,
        width_coefficient: float = 1.4,
        depth_coefficient: float = 1.8,
        dropout_rate: float = 0.5
):
    return EfficientNet(image_channels, n_classes, width_coefficient, depth_coefficient, dropout_rate)


def efficientnet_b5(
        image_channels: int,
        n_classes: int,
        width_coefficient: float = 1.6,
        depth_coefficient: float = 2.2,
        dropout_rate: float = 0.5
):
    return EfficientNet(image_channels, n_classes, width_coefficient, depth_coefficient, dropout_rate)


def efficientnet_b6(
        image_channels: int,
        n_classes: int,
        width_coefficient: float = 1.8,
        depth_coefficient: float = 2.6,
        dropout_rate: float = 0.5
):
    return EfficientNet(image_channels, n_classes, width_coefficient, depth_coefficient, dropout_rate)


def efficientnet_b7(
        image_channels: int,
        n_classes: int,
        width_coefficient: float = 2.0,
        depth_coefficient: float = 3.1,
        dropout_rate: float = 0.5
):
    return EfficientNet(image_channels, n_classes, width_coefficient, depth_coefficient, dropout_rate)