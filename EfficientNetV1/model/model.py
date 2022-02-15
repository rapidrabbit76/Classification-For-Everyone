import torch
import torch.nn as nn
import numpy as np
from .block import ConvBlock, MBConvBlock, Classifier


MODEL_TYPE = {
    # numLayer, input, output, expand ratio, kernel size, padding, stride, reduction ratio
    'MBConv1': [1, 32, 16, 1.0, 3, 1, 1, 4],

    'MBConv6_1': [2, 16, 24, 6.0, 3, 1, 2, 4],

    'MBConv6_2': [2, 24, 40, 6.0, 5, 2, 2, 4],

    'MBConv6_3': [3, 40, 80, 6.0, 3, 1, 2, 4],

    'MBConv6_4': [3, 80, 112, 6.0, 5, 2, 1, 4],

    'MBConv6_5': [4, 112, 192, 6.0, 5, 2, 2, 4],

    'MBConv6_6': [1, 192, 320, 6.0, 3, 1, 1, 4]
}


class EfficientNet(nn.Module):

    def __init__(
            self,
            image_channels: int,
            n_classes: int,
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
            out_features=n_classes,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu',
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def multiply_width(self, dim: int) -> int:
        return int(np.ceil(self.width_coefficient*dim))

    def multiply_depth(self, numLayer: int) -> int:
        return int(np.ceil(self.depth_coefficient*numLayer))