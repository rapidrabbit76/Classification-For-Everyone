from turtle import forward
import torch
from torch import nn

from .blocks import AuxClassifier, Inceptionx2, Inceptionx3, Inceptionx5, ConvBlock, GridReduction


class Inception_v3(nn.Module):

    def __init__(
        self,
        image_channels: int,
        num_classes: int,
        norm: bool,
    ) -> None:
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels=image_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=0,
            norm=norm,
        )

        self.conv2 = ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
            norm=norm,
        )

        self.conv3 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=norm,
        )

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = ConvBlock(
            in_channels=64,
            out_channels=80,
            kernel_size=3,
            stride=1,
            padding=0,
            norm=norm,
        )

        self.conv5 = ConvBlock(
            in_channels=80,
            out_channels=192,
            kernel_size=3,
            stride=2,
            padding=0,
            norm=norm,
        )

        self.conv6 = ConvBlock(
            in_channels=192,
            out_channels=288,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=norm,
        )

        self.inceptionx3 = nn.Sequential(
            Inceptionx3(in_channels=288, dims=[96, 96, 96, 96], norm=norm),
            Inceptionx3(in_channels=96 * 4, dims=[96, 96, 96, 96], norm=norm),
            Inceptionx3(in_channels=96 * 4, dims=[96, 96, 96, 96], norm=norm),
            GridReduction(in_channels=96 * 4, out_channels=96 * 4, norm=norm),
        )
        self.aux = AuxClassifier(
            in_channels=96 * 4 * 2,
            num_classes=num_classes,
            norm=norm,
        )

        self.inceptionx5 = nn.Sequential(
            Inceptionx5(in_channels=96 * 4 * 2, dims=[160] * 4, norm=norm),
            Inceptionx5(in_channels=160 * 4, dims=[160] * 4, norm=norm),
            Inceptionx5(in_channels=160 * 4, dims=[160] * 4, norm=norm),
            Inceptionx5(in_channels=160 * 4, dims=[160] * 4, norm=norm),
            Inceptionx5(in_channels=160 * 4, dims=[160] * 4, norm=norm),
            GridReduction(in_channels=160 * 4, out_channels=160 * 4,
                          norm=norm),
        )

        inceptionx2_dims = [
            [256, 256, 192, 192, 64, 64],
            [384, 384, 384, 384, 256, 256],
        ]
        self.inceptionx2 = nn.Sequential(
            Inceptionx2(
                in_channels=160 * 4 * 2,
                dims=inceptionx2_dims[0],
                norm=norm,
            ),
            Inceptionx2(
                in_channels=sum(inceptionx2_dims[0]),
                dims=inceptionx2_dims[1],
                norm=norm,
            ),
        )

        self.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sum(inceptionx2_dims[1]), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.inceptionx3(x)
        aux = self.aux(x)
        x = self.inceptionx5(x)
        x = self.inceptionx2(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        if self.training:
            return [x, aux]

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
