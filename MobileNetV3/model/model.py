import torch
import torch.nn as nn
import numpy as np
from .block import ConvBlock, BNeckBlock, Classifier


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

        if model_size == '_large':
            model_type = [
                [16, 16, 1.0, 3, 1, 1, 'RE', 3, False],
                [16, 24, 4.0, 3, 1, 2, 'RE', 3, False],
                [24, 24, 3.0, 3, 1, 1, 'RE', 3, False],
                [24, 40, 3.0, 5, 2, 2, 'RE', 3, True],
                [40, 40, 3.0, 5, 2, 1, 'RE', 3, True],
                [40, 40, 3.0, 5, 2, 1, 'RE', 3, True],
                [40, 80, 6.0, 3, 1, 2, 'HS', 3, False],
                [80, 80, 2.4, 3, 1, 1, 'HS', 3, False],
                [80, 80, 2.3, 3, 1, 1, 'HS', 3, False],
                [80, 80, 2.3, 3, 1, 1, 'HS', 3, False],
                [80, 112, 6.0, 3, 1, 1, 'HS', 3, True],
                [112, 112, 6.0, 3, 1, 1, 'HS', 3, True],
                [112, 160, 6.0, 5, 2, 2, 'HS', 3, True],
                [160, 160, 6.0, 5, 2, 1, 'HS', 3, True],
                [160, 160, 6.0, 5, 2, 1, 'HS', 3, True],
                [160, 960, 1],
                [1],
                [960, 1280, 1]
            ]
        elif model_size == '_small':
            model_type = [
                [16, 16, 1.0, 3, 1, 2, 'RE', 3, True],
                [16, 24, 6.0, 3, 1, 2, 'RE', 3, False],
                [24, 24, 3.7, 3, 1, 1, 'RE', 3, False],
                [24, 40, 4.0, 5, 2, 2, 'HS', 3, True],
                [40, 40, 6.0, 5, 2, 1, 'HS', 3, True],
                [40, 40, 6.0, 5, 2, 1, 'HS', 3, True],
                [40, 48, 3.0, 5, 2, 1, 'HS', 3, True],
                [48, 48, 3.0, 5, 2, 1, 'HS', 3, True],
                [48, 96, 6.0, 5, 2, 2, 'HS', 3, True],
                [96, 96, 6.0, 5, 2, 1, 'HS', 3, True],
                [96, 96, 6.0, 5, 2, 1, 'HS', 3, True],
                [96, 576, 1],
                [1],
                [576, 1024, 1]
            ]

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

        for idx in range(len(model_type)):
            if idx <= bneck_size:
                layers.append(
                    BNeckBlock(
                        dim=[
                            self.multiply_width(model_type[idx][0]),
                            self.multiply_width(model_type[idx][1])
                        ],
                        factor=model_type[idx][2],
                        kernel=model_type[idx][3],
                        padding=model_type[idx][4],
                        stride=model_type[idx][5],
                        act=model_type[idx][6],
                        reduction_ratio=model_type[idx][7],
                        use_se=model_type[idx][8]
                    )
                )
            elif idx == bneck_size + 1:
                layers.append(
                    ConvBlock(
                        in_channels=model_type[idx][0],
                        out_channels=model_type[idx][1],
                        kernel_size=model_type[idx][2],
                    )
                ),
                layers.append(
                    nn.BatchNorm2d(
                        num_features=self.multiply_width((model_type[idx][1]))
                    )
                ),
                layers.append(
                    nn.Hardswish()
                ),
                layers.append(nn.AdaptiveAvgPool2d(model_type[idx+1][0]))
            elif idx == bneck_size + 3:
                layers.append(
                    ConvBlock(
                        in_channels=model_type[idx][0],
                        out_channels=model_type[idx][1],
                        kernel_size=model_type[idx][2],
                    )
                ),
                layers.append(
                    nn.Hardswish()
                )

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = Classifier(
            in_features=self.multiply_width(model_type[-1][1]),
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