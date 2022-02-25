import torch
import torch.nn as nn
import numpy as np
from .block import ConvBlock, BNeckBlock, Classifier


MODEL_TYPE = {
    '_large': {
        0: [16, 16, 1.0, 3, 1, 1, 'RE', 3, False],
        1: [16, 24, 4.0, 3, 1, 2, 'RE', 3, False],
        2: [24, 24, 3.0, 3, 1, 1, 'RE', 3, False],
        3: [24, 40, 3.0, 5, 2, 2, 'RE', 3, True],
        4: [40, 40, 3.0, 5, 2, 1, 'RE', 3, True],
        5: [40, 40, 3.0, 5, 2, 1, 'RE', 3, True],
        6: [40, 80, 6.0, 3, 1, 2, 'HS', 3, False],
        7: [80, 80, 2.4, 3, 1, 1, 'HS', 3, False],
        8: [80, 80, 2.3, 3, 1, 1, 'HS', 3, False],
        9: [80, 80, 2.3, 3, 1, 1, 'HS', 3, False],
        10: [80, 112, 6.0, 3, 1, 1, 'HS', 3, True],
        11: [112, 112, 6.0, 3, 1, 1, 'HS', 3, True],
        12: [112, 160, 6.0, 5, 2, 2, 'HS', 3, True],
        13: [160, 160, 6.0, 5, 2, 1, 'HS', 3, True],
        14: [160, 160, 6.0, 5, 2, 1, 'HS', 3, True],
        15: [160, 960, 1],
        16: [1],
        17: [960, 1280, 1]
    },
    '_small': {
        0: [16, 16, 1.0, 3, 1, 2, 'RE', 3, True],
        1: [16, 24, 6.0, 3, 1, 2, 'RE', 3, False],
        2: [24, 24, 3.7, 3, 1, 1, 'RE', 3, False],
        3: [24, 40, 4.0, 5, 2, 2, 'HS', 3, True],
        4: [40, 40, 6.0, 5, 2, 1, 'HS', 3, True],
        5: [40, 40, 6.0, 5, 2, 1, 'HS', 3, True],
        6: [40, 48, 3.0, 5, 2, 1, 'HS', 3, True],
        7: [48, 48, 3.0, 5, 2, 1, 'HS', 3, True],
        8: [48, 96, 6.0, 5, 2, 2, 'HS', 3, True],
        9: [96, 96, 6.0, 5, 2, 1, 'HS', 3, True],
        10: [96, 96, 6.0, 5, 2, 1, 'HS', 3, True],
        11: [96, 576, 1],
        12: [1],
        13: [576, 1024, 1]
    }
}


class MobileNetV3(nn.Module):

    def __init__(
            self,
            image_channels: int,
            n_classes: int,
            alpha: float = 1.0,
            model_size: str = '_large',
            bneck_size: int = 14,
            dropout_rate: float = 0.5
    ) -> None:
        super().__init__()
        self.alpha = alpha
        max_bneck_size = len(MODEL_TYPE[model_size])

        layers = []

        layers.append(
            ConvBlock(
                in_channels=image_channels,
                out_channels=self.multiply_width(16),
                kernel_size=3,
                stride=2,
                padding=1,
            )
        ),
        layers.append(
            nn.BatchNorm2d(
                num_features=self.multiply_width(16)
            )
        ),
        layers.append(
            nn.Hardswish()
        ),

        for idx in range(max_bneck_size):
            if idx <= bneck_size:
                layers.append(
                    BNeckBlock(
                        dim=[
                            self.multiply_width(MODEL_TYPE[model_size][idx][0]),
                            self.multiply_width(MODEL_TYPE[model_size][idx][1])
                        ],
                        factor=MODEL_TYPE[model_size][idx][2],
                        kernel=MODEL_TYPE[model_size][idx][3],
                        padding=MODEL_TYPE[model_size][idx][4],
                        stride=MODEL_TYPE[model_size][idx][5],
                        act=MODEL_TYPE[model_size][idx][6],
                        reduction_ratio=MODEL_TYPE[model_size][idx][7],
                        use_se=MODEL_TYPE[model_size][idx][8]
                    )
                )
            elif idx == bneck_size + 1:
                layers.append(
                    ConvBlock(
                        in_channels=MODEL_TYPE[model_size][idx][0],
                        out_channels=MODEL_TYPE[model_size][idx][1],
                        kernel_size=MODEL_TYPE[model_size][idx][2],
                    )
                ),
                layers.append(
                    nn.BatchNorm2d(
                        num_features=self.multiply_width((MODEL_TYPE[model_size][idx][1]))
                    )
                ),
                layers.append(
                    nn.Hardswish()
                ),
                layers.append(nn.AdaptiveAvgPool2d(MODEL_TYPE[model_size][idx+1][0]))
            elif idx == bneck_size + 3:
                layers.append(
                    ConvBlock(
                        in_channels=MODEL_TYPE[model_size][idx][0],
                        out_channels=MODEL_TYPE[model_size][idx][1],
                        kernel_size=MODEL_TYPE[model_size][idx][2],
                    )
                ),
                layers.append(
                    nn.Hardswish()
                )

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = Classifier(
            in_features=self.multiply_width(MODEL_TYPE[model_size][max_bneck_size-1][1]),
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
        return int(np.ceil(self.alpha*dim))
