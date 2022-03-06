from typing import Union, Tuple
from turtle import forward
import torch
from torch import nn
from torch import Tensor

from .blocks import *


MODEL_RETURN_TYPE = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class Inception_v3(nn.Module):
    def __init__(
        self,
        image_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.conv1 = ConvBlock(image_channels, 32, 3, 2, 0)
        self.conv2 = ConvBlock(32, 32, 3, 1, 0)
        self.conv3 = ConvBlock(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv4 = ConvBlock(64, 80, 3, 1, 0)
        self.conv5 = ConvBlock(80, 192, 3, 2, 0)
        self.conv6 = ConvBlock(192, 288, 3, 1, 1)

        self.inceptionx3 = nn.Sequential(
            Inceptionx3(inp=288, dims=[96] * 4),
            Inceptionx3(inp=96 * 4, dims=[96] * 4),
            Inceptionx3(inp=96 * 4, dims=[96] * 4),
            GridReduction(inp=96 * 4, outp=96 * 4),
        )
        self.aux = AuxClassifier(
            inp=96 * 4 * 2,
            outp=num_classes,
        )

        self.inceptionx5 = nn.Sequential(
            Inceptionx5(inp=96 * 4 * 2, dims=[160] * 4),
            Inceptionx5(inp=160 * 4, dims=[160] * 4),
            Inceptionx5(inp=160 * 4, dims=[160] * 4),
            Inceptionx5(inp=160 * 4, dims=[160] * 4),
            Inceptionx5(inp=160 * 4, dims=[160] * 4),
            GridReduction(inp=160 * 4, outp=160 * 4),
        )

        inceptionx2_dims = [
            [256, 256, 192, 192, 64, 64],
            [384, 384, 384, 384, 256, 256],
        ]
        self.inceptionx2 = nn.Sequential(
            Inceptionx2(
                in_channels=160 * 4 * 2,
                dims=inceptionx2_dims[0],
            ),
            Inceptionx2(
                in_channels=sum(inceptionx2_dims[0]),
                dims=inceptionx2_dims[1],
            ),
        )

        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sum(inceptionx2_dims[1]), num_classes),
        )

    def forward(self, x: Tensor) -> MODEL_RETURN_TYPE:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.inceptionx3(x)
        aux = self.aux(x)
        x = self.inceptionx5(x)
        x = self.inceptionx2(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        # AUX
        return [x, aux] if self.training else x
