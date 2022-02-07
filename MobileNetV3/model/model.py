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
    ) -> None:
        super().__init__()
        self.alpha = alpha

        self.feature_extractor = nn.Sequential(
            ConvBlock(
                in_channels=image_channels,
                out_channels=self.multiply_width(16),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.multiply_width(16)),
            nn.Hardswish(),
            BNeckBlock(
                dim=[
                    self.multiply_width(16),
                    self.multiply_width(16),
                    self.multiply_width(8)
                ],
                factor=1.0,
                kernel=3,
                padding=1,
                stride=1,
                act='RE',
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(16),
                    self.multiply_width(24),
                    self.multiply_width(12)
                ],
                factor=4.0,
                kernel=3,
                padding=1,
                stride=2,
                act='RE',
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(24),
                    self.multiply_width(24),
                    self.multiply_width(12)
                ],
                factor=3.0,
                kernel=3,
                padding=1,
                stride=1,
                act='RE',
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(24),
                    self.multiply_width(40),
                    self.multiply_width(20)
                ],
                factor=3.0,
                kernel=5,
                padding=2,
                stride=2,
                act='RE',
                use_se=True,
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(40),
                    self.multiply_width(40),
                    self.multiply_width(20)
                ],
                factor=3.0,
                kernel=5,
                padding=2,
                stride=1,
                act='RE',
                use_se=True,
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(40),
                    self.multiply_width(40),
                    self.multiply_width(20)
                ],
                factor=3.0,
                kernel=5,
                padding=2,
                stride=1,
                act='RE',
                use_se=True,
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(40),
                    self.multiply_width(80),
                    self.multiply_width(40)
                ],
                factor=6.0,
                kernel=3,
                padding=1,
                stride=2,
                act='HS',
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(80),
                    self.multiply_width(80),
                    self.multiply_width(40)
                ],
                factor=2.4,
                kernel=3,
                padding=1,
                stride=1,
                act='HS',
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(80),
                    self.multiply_width(80),
                    self.multiply_width(40)
                ],
                factor=2.3,
                kernel=3,
                padding=1,
                stride=1,
                act='HS',
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(80),
                    self.multiply_width(80),
                    self.multiply_width(40)
                ],
                factor=2.3,
                kernel=3,
                padding=1,
                stride=1,
                act='HS',
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(80),
                    self.multiply_width(112),
                    self.multiply_width(56)
                ],
                factor=6.0,
                kernel=3,
                padding=1,
                stride=1,
                act='HS',
                use_se=True,
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(112),
                    self.multiply_width(112),
                    self.multiply_width(56)
                ],
                factor=6.0,
                kernel=3,
                padding=1,
                stride=1,
                act='HS',
                use_se=True,
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(112),
                    self.multiply_width(160),
                    self.multiply_width(80)
                ],
                factor=6.0,
                kernel=5,
                padding=2,
                stride=2,
                act='HS',
                use_se=True,
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(160),
                    self.multiply_width(160),
                    self.multiply_width(80)
                ],
                factor=6.0,
                kernel=5,
                padding=2,
                stride=1,
                act='HS',
                use_se=True,
            ),
            BNeckBlock(
                dim=[
                    self.multiply_width(160),
                    self.multiply_width(160),
                    self.multiply_width(80)
                ],
                factor=6.0,
                kernel=5,
                padding=2,
                stride=1,
                act='HS',
                use_se=True,
            ),
            ConvBlock(
                in_channels=self.multiply_width(160),
                out_channels=self.multiply_width(960),
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=self.multiply_width(960)),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(
                in_channels=self.multiply_width(960),
                out_channels=self.multiply_width(1280),
                kernel_size=1,
            ),
            nn.Hardswish(),
            ConvBlock(
                in_channels=self.multiply_width(1280),
                out_channels=self.multiply_width(1000),
                kernel_size=1,
            ),
        )
        self.classifier = Classifier(
            in_features=self.multiply_width(1000),
            out_features=n_classes
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