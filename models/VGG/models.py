from typing import Final, Dict
import torch
from torch import nn

from .blocks import Classifier, ConvBlock

__all__ = ["VGGModel", "VGG11", "VGG13", "VGG16", "VGG19"]

MODEL_TYPES: Final[Dict] = {
    # fmt: off
    "VGG11": [64,  "M", 128, "M", 256, 256, "M", 512, 512, "M", 
              512, 512, "M"],
    
    "VGG13": [64,   64, "M", 128, 128, "M", 256, 256, "M", 
              512, 512, "M", 512, 512, "M"],
    
    "VGG16": [64,   64, "M", 128, 128, "M", 256, 256, 256, "M", 
              512, 512, 512, "M", 512, 512, 512, "M"],
    
    "VGG19": [64,   64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 
              512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGGModel(nn.Module):
    def __init__(
        self,
        model_type: str = "VGG11",
        image_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: int = 0.5,
    ) -> None:
        super().__init__()
        layers = []

        assert (
            model_type in MODEL_TYPES.keys()
        ), f"{model_type} is not in {' '.join(MODEL_TYPES.keys())}"

        in_channels = image_channels
        for x in MODEL_TYPES[model_type]:
            if type(x) == int:
                layers.append(ConvBlock(in_channels, x))
                in_channels = x
            elif x == "M":
                layers.append(nn.MaxPool2d(2, 2))

        self.feature_extractor = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classfier = nn.Sequential(
            nn.Flatten(),
            Classifier(512 * 7 * 7, 4096, num_classes, dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        return self.classfier(x)


def VGG11(
    image_channals: int,
    num_classes: int,
    dropout_rate: int = 0.5,
):
    return VGGModel("VGG11", image_channals, num_classes, dropout_rate)


def VGG13(
    image_channals: int,
    num_classes: int,
    dropout_rate: int = 0.5,
):
    return VGGModel("VGG13", image_channals, num_classes, dropout_rate)


def VGG16(
    image_channals: int,
    num_classes: int,
    dropout_rate: int = 0.5,
):
    return VGGModel("VGG16", image_channals, num_classes, dropout_rate)


def VGG19(
    image_channals: int,
    num_classes: int,
    dropout_rate: int = 0.5,
):
    return VGGModel("VGG19", image_channals, num_classes, dropout_rate)
