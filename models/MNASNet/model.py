import torch
import torch.nn as nn
import numpy as np
from .block import ConvBlock, SepConvBlock, MBConvBlock, Classifier


MODEL_TYPE = {
    'SepConv': {
        0: [32, 16, 1.0, 1, 1],
    },
    'MBConv6_1': {
        0: [16, 16, 6.0, 3, 1, 1, 3, False],
        1: [16, 24, 6.0, 3, 1, 2, 3, False],
    },
    'MBConv3CE_1': {
        0: [24, 24, 3.0, 5, 2, 1, 3, True],
        1: [24, 24, 3.0, 5, 2, 1, 3, True],
        2: [24, 40, 3.0, 5, 2, 2, 3, True],
    },
    'MBConv6_2': {
        0: [40, 40, 6.0, 3, 1, 1, 3, False],
        1: [40, 40, 6.0, 3, 1, 1, 3, False],
        2: [40, 40, 6.0, 3, 1, 1, 3, False],
        3: [40, 80, 6.0, 3, 1, 2, 3, False],
    },
    'MBConv6SE_1': {
        0: [80, 80, 6.0, 3, 1, 1, 3, True],
        1: [80, 112, 6.0, 3, 1, 1, 3, True],
    },
    'MBConv6SE_2': {
        0: [112, 112, 6.0, 5, 2, 1, 3, True],
        1: [112, 112, 6.0, 5, 2, 1, 3, True],
        2: [112, 160, 6.0, 5, 2, 2, 3, True],
    },
    'MBConv6_3': {
        0: [160, 320, 6.0, 3, 1, 1, 3, False],
    },
}


class MNASNet(nn.Module):

    def __init__(
            self,
            image_channels: int,
            n_classes: int,
            alpha: float = 1.0,
            dropout_rate: float = 0.5
    ) -> None:
        super().__init__()
        self.alpha = alpha
        blockKeys = MODEL_TYPE.keys()
        layers = []

        layers.append(
            ConvBlock(
                in_channels=image_channels,
                out_channels=self.multiply_width(32),
                kernel_size=3,
                stride=2,
                padding=1,
            )
        ),
        layers.append(
            nn.BatchNorm2d(
                num_features=self.multiply_width(32)
            )
        ),
        layers.append(
            nn.ReLU()
        ),

        for key in blockKeys:
            for block_idx in MODEL_TYPE[key]:
                if key == 'SepConv':
                    layers.append(
                        SepConvBlock(
                            dim=[
                                self.multiply_width(MODEL_TYPE[key][block_idx][0]),
                                self.multiply_width(MODEL_TYPE[key][block_idx][1])
                            ],
                            factor=MODEL_TYPE[key][block_idx][2],
                            stride=MODEL_TYPE[key][block_idx][3],
                            padding=MODEL_TYPE[key][block_idx][4],
                        )
                    )
                else:
                    layers.append(
                        MBConvBlock(
                            dim=[
                                self.multiply_width(MODEL_TYPE[key][block_idx][0]),
                                self.multiply_width(MODEL_TYPE[key][block_idx][1])
                            ],
                            factor=MODEL_TYPE[key][block_idx][2],
                            kernel=MODEL_TYPE[key][block_idx][3],
                            padding=MODEL_TYPE[key][block_idx][4],
                            stride=MODEL_TYPE[key][block_idx][5],
                            reduction_ratio=MODEL_TYPE[key][block_idx][6],
                            use_se=MODEL_TYPE[key][block_idx][7]
                        )
                    )

        layers.append(
            ConvBlock(
                in_channels=self.multiply_width(MODEL_TYPE['MBConv6_3'][0][1]),
                out_channels=self.multiply_width(1280),
                kernel_size=1,
            )
        ),
        layers.append(
            nn.BatchNorm2d(
                num_features=self.multiply_width(1280)
            )
        ),
        layers.append(
            nn.ReLU()
        ),

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
        return int(np.ceil(self.alpha*dim))
