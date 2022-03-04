import torch
import torch.nn as nn
import numpy as np
from .blocks import *

__all__ = [
    "MobileNetV3", "mobilenetv3_large", "mobilenetv3_small"
]


MODEL_TYPE = {
    # input, output, expansion ratio, kernel size, padding, stride,
    # activation, reduction ratio, use SEBlock or not
    # fmt: off
    'large': {
        0:  [16,   16, 1.0, 3, 1, 1, 'RE', 2, False],
        1:  [16,   24, 4.0, 3, 1, 2, 'RE', 2, False],
        2:  [24,   24, 3.0, 3, 1, 1, 'RE', 2, False],
        3:  [24,   40, 3.0, 5, 2, 2, 'RE', 2, True],
        4:  [40,   40, 3.0, 5, 2, 1, 'RE', 2, True],
        5:  [40,   40, 3.0, 5, 2, 1, 'RE', 2, True],
        6:  [40,   80, 6.0, 3, 1, 2, 'HS', 2, False],
        7:  [80,   80, 2.4, 3, 1, 1, 'HS', 2, False],
        8:  [80,   80, 2.3, 3, 1, 1, 'HS', 2, False],
        9:  [80,   80, 2.3, 3, 1, 1, 'HS', 2, False],
        10: [80,  112, 6.0, 3, 1, 1, 'HS', 2, True],
        11: [112, 112, 6.0, 3, 1, 1, 'HS', 2, True],
        12: [112, 160, 6.0, 5, 2, 2, 'HS', 2, True],
        13: [160, 160, 6.0, 5, 2, 1, 'HS', 2, True],
        14: [160, 160, 6.0, 5, 2, 1, 'HS', 2, True],
        15: [160, 960, 1],
        16: [1],
        17: [960, 1280, 1]
    },
    # fmt: off
    'small': {
        0:  [16, 16, 1.0, 3, 1, 2, 'RE', 2, True],
        1:  [16, 24, 6.0, 3, 1, 2, 'RE', 2, False],
        2:  [24, 24, 3.7, 3, 1, 1, 'RE', 2, False],
        3:  [24, 40, 4.0, 5, 2, 2, 'HS', 2, True],
        4:  [40, 40, 6.0, 5, 2, 1, 'HS', 2, True],
        5:  [40, 40, 6.0, 5, 2, 1, 'HS', 2, True],
        6:  [40, 48, 3.0, 5, 2, 1, 'HS', 2, True],
        7:  [48, 48, 3.0, 5, 2, 1, 'HS', 2, True],
        8:  [48, 96, 6.0, 5, 2, 2, 'HS', 2, True],
        9:  [96, 96, 6.0, 5, 2, 1, 'HS', 2, True],
        10: [96, 96, 6.0, 5, 2, 1, 'HS', 2, True],
        11: [96, 576, 1],
        12: [1],
        13: [576, 1024, 1]
    }
}


class MobileNetV3(nn.Module):

    def __init__(
            self,
            image_channels: int,
            num_classes: int,
            alpha: float = 1.0,
            model_type: str = 'large',
            dropout_rate: float = 0.5
    ) -> None:
        super().__init__()
        self.alpha = alpha
        max_bneck_size = len(MODEL_TYPE[model_type])
        bneck_size = max_bneck_size - 4
        layers = []

        layers.append(
            ConvBlock(
                in_channels=image_channels,
                out_channels=self.multiply_width(16),
                kernel_size=3,
                stride=2,
                padding=1,
                act='HS'
            )
        ),

        for idx in range(max_bneck_size):
            if idx <= bneck_size:
                layers.append(
                    BNeckBlock(
                        dim=[
                            self.multiply_width(MODEL_TYPE[model_type][idx][0]),
                            self.multiply_width(MODEL_TYPE[model_type][idx][1])
                        ],
                        factor=MODEL_TYPE[model_type][idx][2],
                        kernel=MODEL_TYPE[model_type][idx][3],
                        padding=MODEL_TYPE[model_type][idx][4],
                        stride=MODEL_TYPE[model_type][idx][5],
                        act=MODEL_TYPE[model_type][idx][6],
                        reduction_ratio=MODEL_TYPE[model_type][idx][7],
                        use_se=MODEL_TYPE[model_type][idx][8]
                    )
                )
            elif idx == bneck_size + 1:
                layers.append(
                    ConvBlock(
                        in_channels=MODEL_TYPE[model_type][idx][0],
                        out_channels=MODEL_TYPE[model_type][idx][1],
                        kernel_size=MODEL_TYPE[model_type][idx][2],
                        act='HS',
                    )
                ),
                layers.append(nn.AdaptiveAvgPool2d(MODEL_TYPE[model_type][idx+1][0]))
            elif idx == bneck_size + 3:
                layers.append(
                    ConvBlock(
                        in_channels=MODEL_TYPE[model_type][idx][0],
                        out_channels=MODEL_TYPE[model_type][idx][1],
                        kernel_size=MODEL_TYPE[model_type][idx][2],
                        act='HS',
                    )
                ),

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = Classifier(
            in_features=self.multiply_width(MODEL_TYPE[model_type][max_bneck_size-1][1]),
            out_features=num_classes,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def multiply_width(self, dim: int) -> int:
        return int(np.ceil(self.alpha*dim))


def mobilenetv3_large(
        image_channels: int,
        num_classes: int,
        alpha: float = 1.0,
        model_type: str = 'large',
        dropout_rate: float = 0.5
) -> MobileNetV3:
    return MobileNetV3(image_channels, num_classes, alpha, model_type, dropout_rate)


def mobilenetv3_small(
        image_channels: int,
        num_classes: int,
        alpha: float = 1.0,
        model_type: str = 'small',
        dropout_rate: float = 0.5
) -> MobileNetV3:
    return MobileNetV3(image_channels, num_classes, alpha, model_type, dropout_rate)