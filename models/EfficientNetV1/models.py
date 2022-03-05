import torch
import torch.nn as nn
import numpy as np
from .blocks import *
from typing import Final, Dict

__all__ = [
    "EfficientNet",
    "EfficientNet_b0",
    "EfficientNet_b1",
    "EfficientNet_b2",
    "EfficientNet_b3",
    "EfficientNet_b4",
    "EfficientNet_b5",
    "EfficientNet_b6",
    "EfficientNet_b7",
]

MODEL_BLOCKS: Final[Dict] = {
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

MODEL_TYPES: Final[Dict] = {
    # width depth image size
    # fmt: off
    'b0': [1.0, 1.0, 224],
    'b1': [1.0, 1.1, 240],
    'b2': [1.1, 1.2, 260],
    'b3': [1.2, 1.4, 300],
    'b4': [1.4, 1.8, 380],
    'b5': [1.6, 2.2, 456],
    'b6': [1.8, 2.6, 528],
    'b7': [2.0, 3.1, 600]
}


class EfficientNet(nn.Module):
    def __init__(
        self,
        model_type: str,
        image_channels: int,
        num_classes: int,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        layers = []
        Blockkeys = MODEL_BLOCKS.keys()
        model_types = MODEL_TYPES[model_type]
        self.width_coefficient = model_types[0]
        self.depth_coefficient = model_types[1]

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
                    numLayer=self.multiply_depth(numLayer=MODEL_BLOCKS[key][0]),
                    dim=[
                        self.multiply_width(MODEL_BLOCKS[key][1]),
                        self.multiply_width(MODEL_BLOCKS[key][2]),
                    ],
                    factor=MODEL_BLOCKS[key][3],
                    kernel_size=MODEL_BLOCKS[key][4],
                    padding=MODEL_BLOCKS[key][5],
                    stride=MODEL_BLOCKS[key][6],
                    reduction_ratio=MODEL_BLOCKS[key][7],
                )
            )

        layers.append(
            ConvBlock(
                in_channels=self.multiply_width(MODEL_BLOCKS["MBConv6_6"][2]),
                out_channels=self.multiply_width(1280),
                kernel_size=1,
            )
        )

        layers.append(nn.AdaptiveAvgPool2d(1))

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = Classifier(
            in_features=self.multiply_width(1280),
            out_features=num_classes,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def multiply_width(self, dim: int) -> int:
        return int(np.ceil(self.width_coefficient * dim))

    def multiply_depth(self, numLayer: int) -> int:
        return int(np.ceil(self.depth_coefficient * numLayer))


def EfficientNet_b0(
    image_channels: int, num_classes: int, dropout_rate: float = 0.5
) -> EfficientNet:
    return EfficientNet("b0", image_channels, num_classes, dropout_rate)


def EfficientNet_b1(
    image_channels: int, num_classes: int, dropout_rate: float = 0.5
) -> EfficientNet:
    return EfficientNet("b1", image_channels, num_classes, dropout_rate)


def EfficientNet_b2(
    image_channels: int, num_classes: int, dropout_rate: float = 0.5
) -> EfficientNet:
    return EfficientNet("b2", image_channels, num_classes, dropout_rate)


def EfficientNet_b3(
    image_channels: int, num_classes: int, dropout_rate: float = 0.5
) -> EfficientNet:
    return EfficientNet("b3", image_channels, num_classes, dropout_rate)


def EfficientNet_b4(
    image_channels: int, num_classes: int, dropout_rate: float = 0.5
) -> EfficientNet:
    return EfficientNet("b4", image_channels, num_classes, dropout_rate)


def EfficientNet_b5(
    image_channels: int, num_classes: int, dropout_rate: float = 0.5
) -> EfficientNet:
    return EfficientNet("b5", image_channels, num_classes, dropout_rate)


def EfficientNet_b6(
    image_channels: int, num_classes: int, dropout_rate: float = 0.5
) -> EfficientNet:
    return EfficientNet("b6", image_channels, num_classes, dropout_rate)


def EfficientNet_b7(
    image_channels: int, num_classes: int, dropout_rate: float = 0.5
) -> EfficientNet:
    return EfficientNet("b7", image_channels, num_classes, dropout_rate)
