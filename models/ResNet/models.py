import torch
import torch.nn as nn
from .blocks import *

__all__ = [
    "ResNet", "ResNet_18", "ResNet_34",
    "ResNet_50", "ResNet_101", "ResNet_152"
]


RESNET_TYPE = {
    '18': [2, 2, 2, 2],
    '34': [3, 4, 6, 3],
    '50': [3, 4, 6, 3],
    '101': [3, 4, 23, 3],
    '152': [3, 8, 36, 3],
}


class ResNet(nn.Module):

    def __init__(
            self,
            model_type: str,
            image_channels: int,
            num_classes: int,
            dropout_rate: float = 0.5
    ):
        super().__init__()
        dim = 64
        num_layers = int(model_type)
        layers = []
        layers += [ConvBlock(image_channels, dim, kernel_size=7, stride=2, padding=3)]
        layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        # stack blocks
        listBlocks = RESNET_TYPE[model_type]
        for idx_layer, nblock in enumerate(listBlocks):
            layers += [ResidualBlock(num_layers, idx_layer, nblock, dim)]
            dim *= 2

        layers += [nn.AdaptiveAvgPool2d(1)]

        if num_layers < 50:
            dim = dim // 2
        else:
            dim = dim * 2

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = Classifier(
            in_features=int(dim),
            out_features=num_classes,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def ResNet_18(
        image_channels: int,
        num_classes: int,
        dropout_rate: float = 0.5
) -> ResNet:
    return ResNet('18', image_channels, num_classes, dropout_rate)


def ResNet_34(
        image_channels: int,
        num_classes: int,
        dropout_rate: float = 0.5
) -> ResNet:
    return ResNet('34', image_channels, num_classes, dropout_rate)


def ResNet_50(
        image_channels: int,
        num_classes: int,
        dropout_rate: float = 0.5
) -> ResNet:
    return ResNet('50', image_channels, num_classes, dropout_rate)


def ResNet_101(
        image_channels: int,
        num_classes: int,
        dropout_rate: float = 0.5
) -> ResNet:
    return ResNet('101', image_channels, num_classes, dropout_rate)


def ResNet_152(
        image_channels: int,
        num_classes: int,
        dropout_rate: float = 0.5
) -> ResNet:
    return ResNet('152', image_channels, num_classes, dropout_rate)