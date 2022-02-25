import torch
import torch.nn as nn
from .block import ConvBlock, DepthWiseSeparableBlock, Classifier


class MobileNetV1(nn.Module):

    def __init__(
            self,
            image_channels: int,
            n_classes: int,
            alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(
                in_channels=image_channels,
                out_channels=int(alpha*32),
                kernel_size=3,
                stride=2,
                padding=1
            ),
            DepthWiseSeparableBlock(
                dim=[int(alpha*32), int(alpha*64), int(alpha*128)],
                stride=[1, 1, 2, 1],
                is_common=True
            ),
            DepthWiseSeparableBlock(
                dim=[int(alpha*128), int(alpha*128), int(alpha*256)],
                stride=[1, 1, 2, 1],
                is_common=True
            ),
            DepthWiseSeparableBlock(
                dim=[int(alpha*256), int(alpha*256), int(alpha*512)],
                stride=[1, 1, 2, 1],
                is_common=True
            ),
            DepthWiseSeparableBlock(
                dim=[int(alpha*512)],
                stride=[1],
                iterate=5,
                is_repeat=True
            ),
            DepthWiseSeparableBlock(
                dim=[int(alpha*512), int(alpha*1024), int(alpha*1024)],
                stride=[2, 1, 2, 1],
                is_last=True
            ),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = Classifier(in_features=int(alpha*1024), out_features=n_classes)

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