from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *


__all__ = [
    "ShuffleNetV2",
    "ShuffleNetV2_x05",
    "ShuffleNetV2_x10",
    "ShuffleNetV2_x15",
    "ShuffleNetV2_x20",
]


class ModelType:
    def __init__(
        self,
        repeat: List[int],
        out_channels: List[int],
    ) -> None:
        self.repeat = repeat
        self.out_channels = out_channels


MODEL_TYPES: Dict[str, ModelType] = {
    "x05": ModelType([4, 8, 4], [24, 48, 96, 192, 1024]),
    "x10": ModelType([4, 8, 4], [24, 116, 232, 464, 1024]),
    "x15": ModelType([4, 8, 4], [24, 176, 352, 704, 1024]),
    "x20": ModelType([4, 8, 4], [24, 244, 488, 976, 2048]),
}


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        model_type: str,
        image_channels: int,
        num_classes: int,
    ):
        super().__init__()
        if model_type not in list(MODEL_TYPES.keys()):
            raise Exception(
                f"{model_type} not supported select one onf {MODEL_TYPES.keys()}"
            )

        model_type = MODEL_TYPES[model_type]

        repeat = model_type.repeat
        out_channels = model_type.out_channels

        self.conv1 = ConvBlock(image_channels, out_channels[0], 3, 2, 1)
        self.max_pool = nn.MaxPool2d(3, 2, 1)

        # build model
        self.stage2 = self._make_layer(out_channels[0], out_channels[1], repeat[0])
        self.stage3 = self._make_layer(out_channels[1], out_channels[2], repeat[1])
        self.stage4 = self._make_layer(out_channels[2], out_channels[3], repeat[2])

        self.conv2 = ConvBlock(out_channels[3], out_channels[-1], 1)
        self.classifier = Classifier(out_channels[-1], num_classes)

    @classmethod
    def _make_layer(cls, inp: int, outp: int, r: int) -> nn.Sequential:
        layer = [ShuffleNetUnit(inp, outp, 2)]
        layer += [ShuffleNetUnit(outp, outp, 1) for _ in range(1, r)]
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


def ShuffleNetV2_x05(image_channels: int, num_classes: int):
    return ShuffleNetV2("x05", image_channels, num_classes)


def ShuffleNetV2_x10(image_channels: int, num_classes: int):
    return ShuffleNetV2("x10", image_channels, num_classes)


def ShuffleNetV2_x15(image_channels: int, num_classes: int):
    return ShuffleNetV2("x15", image_channels, num_classes)


def ShuffleNetV2_x20(image_channels: int, num_classes: int):
    return ShuffleNetV2("x20", image_channels, num_classes)
