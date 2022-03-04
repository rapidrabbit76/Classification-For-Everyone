import torch
import torch.nn as nn
from .blocks import *

__all__ = [
    "ResNet", "resnet_18", "resnet_34",
    "resnet_50", "resnet_101", "resnet_152"
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
            image_channels: int,
            num_classes: int,
            model_type: str,
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

        layers += [nn.AvgPool2d(kernel_size=7)]

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


def resnet_18(
        image_channels: int,
        num_classes: int,
        model_type: str = '18',
        dropout_rate: float = 0.5
) -> ResNet:
    return ResNet(image_channels, num_classes, model_type, dropout_rate)


def resnet_34(
        image_channels: int,
        num_classes: int,
        model_type: str = '34',
        dropout_rate: float = 0.5
) -> ResNet:
    return ResNet(image_channels, num_classes, model_type, dropout_rate)


def resnet_50(
        image_channels: int,
        num_classes: int,
        model_type: str = '50',
        dropout_rate: float = 0.5
) -> ResNet:
    return ResNet(image_channels, num_classes, model_type, dropout_rate)


def resnet_101(
        image_channels: int,
        num_classes: int,
        model_type: str = '101',
        dropout_rate: float = 0.5
) -> ResNet:
    return ResNet(image_channels, num_classes, model_type, dropout_rate)


def resnet_152(
        image_channels: int,
        num_classes: int,
        model_type: str = '152',
        dropout_rate: float = 0.5
) -> ResNet:
    return ResNet(image_channels, num_classes, model_type, dropout_rate)