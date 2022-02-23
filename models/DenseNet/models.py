import torch
from torch import nn
from .blocks import DenseBlock, TransitionBlock

DENSE_NET_TYPE = {
    "121": [6, 12, 24, 16],
    "169": [6, 12, 32, 32],
    "201": [6, 12, 48, 32],
    "265": [6, 12, 64, 48],
}


class DenseNet(nn.Module):
    def __init__(
        self,
        model_type: str,
        image_channels: int,
        nun_classes: int,
        growth_rate: int,
    ) -> None:
        super().__init__()
        layers = []
        dim = growth_rate * 2
        layers += [nn.Conv2d(image_channels, dim, 7, 2, 3, bias=False)]
        layers += [nn.MaxPool2d(3, 2, padding=1)]

        model_type = DENSE_NET_TYPE[model_type]

        for idx, layer in enumerate(model_type):
            layers += [DenseBlock(layer, dim, growth_rate)]
            dim += growth_rate * layer
            if idx == len(model_type) - 1:
                continue
            layers += [TransitionBlock(dim, dim // 2)]
            dim = dim // 2

        self.feature_extractor = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, nun_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        return self.classifier(x)


def DenseNet121(
    image_channels: int,
    nun_classes: int,
    growth_rate: int = 12,
):
    return DenseNet("121", image_channels, nun_classes, growth_rate)


def DenseNet169(
    image_channels: int,
    nun_classes: int,
    growth_rate: int = 12,
):
    return DenseNet("169", image_channels, nun_classes, growth_rate)


def DenseNet201(
    image_channels: int,
    nun_classes: int,
    growth_rate: int = 12,
):
    return DenseNet("201", image_channels, nun_classes, growth_rate)


def DenseNet265(
    image_channels: int,
    nun_classes: int,
    growth_rate: int = 12,
):
    return DenseNet("265", image_channels, nun_classes, growth_rate)
