from math import log
from xml.etree.ElementInclude import include
import torch
from torch import nn

from .blocks import Classifier, ConvBlock

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGGModel(nn.Module):
    def __init__(
        self,
        model_type: str = "VGG11",
        image_channals: int = 3,
        n_classes: int = 1000,
        use_bn: bool = False,
    ) -> None:
        super().__init__()

        layers = []

        assert (
            model_type in VGG_types.keys()
        ), f"{model_type} is not in {' '.join(VGG_types.keys())}"

        in_channels = image_channals
        for x in VGG_types[model_type]:
            if type(x) == int:
                layers.append(ConvBlock(in_channels, x, norm=use_bn))
                in_channels = x
            elif x == "M":
                layers.append(nn.MaxPool2d(2, 2))

        self.feature_extractor = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classfier = nn.Sequential(
            nn.Flatten(),
            Classifier(512 * 7 * 7, 4096, n_classes, 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        return self.classfier(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
