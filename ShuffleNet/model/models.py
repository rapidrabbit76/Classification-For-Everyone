from black import out
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
from typing import List, Union


__all__ = [
    "ShuffleNetV2",
]


class ModelType:
    def __init__(
        self,
        repeat: List[int],
        out_channels: List[int],
    ) -> None:
        self.repeat = repeat
        self.out_chaanels = out_channels


class ModelTypes:
    x05 = ModelType([4, 8, 4], [24, 48, 96, 192, 1024])
    x10 = ModelType([4, 8, 4], [24, 116, 232, 464, 1024])
    x15 = ModelType([4, 8, 4], [24, 176, 352, 704, 1024])
    x20 = ModelType([4, 8, 4], [24, 244, 488, 976, 2048])


MODEL_LIST = dir(ModelTypes)[26:]


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        model_type: str,
        image_channels: int,
        num_classes: int,
    ):
        super().__init__()
        if model_type not in MODEL_LIST:
            print("ERR")
            raise Exception(f"{model_type} not supported sellect one onf {MODEL_LIST}")

        repeat = getattr(ModelTypes, model_type).repeat
        out_chaanels = getattr(ModelTypes, model_type).out_chaanels

        self.conv1 = ConvBlock(
            in_channels=image_channels,
            out_channels=out_chaanels[0],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        in_channels = out_chaanels[0]

        # build model
        self.stage2 = self._make_layer(
            in_channels,
            out_chaanels[1],
            repeat[0],
        )
        in_channels = out_chaanels[1]
        self.stage3 = self._make_layer(
            in_channels,
            out_chaanels[2],
            repeat[1],
        )
        in_channels = out_chaanels[2]
        self.stage4 = self._make_layer(
            in_channels,
            out_chaanels[3],
            repeat[2],
        )

        in_channels = out_chaanels[3]
        self.conv2 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_chaanels[-1],
            kernel_size=1,
        )
        self.classifier = Classifier(
            in_features=out_chaanels[-1],
            num_classes=num_classes,
        )

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        repeat: int,
    ) -> nn.Sequential:
        layer = [ShuffleNetUnit(in_channels, out_channels, 2)]
        in_channels = out_channels
        for _ in range(repeat - 1):
            layer += [ShuffleNetUnit(in_channels, out_channels, 1)]
        return nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv2(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

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
