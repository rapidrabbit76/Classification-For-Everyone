from black import out
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
from typing import List, Union
import inspect

__all__ = [
    "EfficientNetV2",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "efficientnet_v2_lx",
]


class ModelType:
    def __init__(
        self,
        layers: List[int],
        out_channels: List[int],
        strides: List[int],
        expand_ratios: List[int],
        fused: List[int],
    ) -> None:
        # Check config
        assert (
            len(layers)
            == len(out_channels)
            == len(strides)
            == len(expand_ratios)
            == len(fused)
        )

        self.layers = layers
        self.out_channels = out_channels
        self.strides = strides
        self.expand_ratios = expand_ratios
        self.fused = fused


class ModelTypes:
    s = ModelType(
        layers=[2, 4, 4, 6, 9, 15],
        out_channels=[24, 48, 64, 128, 160, 256],
        strides=[1, 2, 2, 2, 1, 2],
        expand_ratios=[1, 4, 4, 4, 6, 6],
        fused=[1, 1, 1, 0, 0, 0],
    )
    m = ModelType(
        layers=[3, 5, 5, 7, 14, 18, 5],
        out_channels=[24, 48, 80, 160, 176, 304, 512],
        strides=[1, 2, 2, 2, 1, 2, 1],
        expand_ratios=[1, 4, 4, 4, 6, 6, 6],
        fused=[1, 1, 1, 0, 0, 0, 0],
    )
    l = ModelType(
        layers=[4, 7, 7, 10, 19, 25, 7],
        out_channels=[32, 64, 96, 192, 224, 384, 640],
        strides=[1, 2, 2, 2, 1, 2, 1],
        expand_ratios=[1, 4, 4, 4, 6, 6, 6],
        fused=[1, 1, 1, 0, 0, 0, 0],
    )
    xl = ModelType(
        layers=[4, 8, 8, 16, 24, 32, 8],
        out_channels=[32, 64, 96, 192, 256, 512, 640],
        strides=[1, 2, 2, 2, 1, 2, 1],
        expand_ratios=[1, 4, 4, 4, 6, 6, 6],
        fused=[1, 1, 1, 0, 0, 0, 0],
    )


MODEL_LIST = dir(ModelTypes)[26:]


class EfficientNetV2(nn.Module):
    def __init__(
        self,
        model_type: str,
        image_channels: int,
        num_classes: int,
    ):
        super().__init__()
        if model_type not in MODEL_LIST:
            raise Exception(f"{model_type} not supported sellect one onf {MODEL_LIST}")

        model_type: ModelType = getattr(ModelTypes, model_type)
        layers = model_type.layers
        out_channels = model_type.out_channels
        strides = model_type.strides
        expand_ratios = model_type.expand_ratios
        fused = model_type.fused

        self.conv1 = ConvBlock(
            in_channels=image_channels,
            out_channels=24,
            kernel_size=3,
            stride=2,
        )
        in_channels = 24
        stage_layers = []
        for idx, layer_count in enumerate(layers):
            stage_layers += [
                self._make_layer(
                    in_channels=in_channels,
                    layer_count=layer_count,
                    dim=out_channels[idx],
                    stride=strides[idx],
                    expand_ratio=expand_ratios[idx],
                    fused=fused[idx],
                )
            ]
            in_channels = out_channels[idx]
        self.stage_layers = nn.Sequential(*stage_layers)

        out_channels = 1792
        self.conv2 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_channels = out_channels
        self.classifier = Classifier(
            in_features=in_channels,
            num_classes=num_classes,
        )

    def _make_layer(
        self,
        in_channels: int,
        dim: int,
        layer_count: int,
        stride: int,
        expand_ratio: int,
        fused: int,
    ) -> nn.Sequential:
        layers = []

        for i in range(layer_count):
            layers += [
                MBConvBlock(
                    in_channels=in_channels,
                    out_channels=dim,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    is_fused=fused,
                )
            ]
            in_channels = dim
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.stage_layers(x)
        x = self.conv2(x)
        x = self.avgpool(x)
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


def efficientnet_v2_s(image_size: int, num_classes: int) -> EfficientNetV2:
    return EfficientNetV2("s", image_size, num_classes)


def efficientnet_v2_m(image_size: int, num_classes: int) -> EfficientNetV2:
    return EfficientNetV2("m", image_size, num_classes)


def efficientnet_v2_l(image_size: int, num_classes: int) -> EfficientNetV2:
    return EfficientNetV2("l", image_size, num_classes)


def efficientnet_v2_lx(image_size: int, num_classes: int) -> EfficientNetV2:
    return EfficientNetV2("lx", image_size, num_classes)
