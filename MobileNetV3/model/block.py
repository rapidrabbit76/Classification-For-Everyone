import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BNeckBlock(nn.Module):

    def __init__(
            self,
            dim: List[int],
            factor: int,
            stride: int,
            act: str,
            use_se: bool = False,
    ) -> None:
        super().__init__()
        self.use_se = use_se
        self.use_res_connection = (stride == 1) and (dim[0] == dim[1])
        self.act = nn.Hardswish() if act == 'HE' else nn.ReLU6()

        self.first_block = nn.Sequential(
            # Conv 1x1
            ConvBlock(
                in_channels=dim[0],
                out_channels=dim[0]*factor,
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=dim[0]*factor),
            self.act,
            # Dwise 3x3
            ConvBlock(
                in_channels=dim[0]*factor,
                out_channels=dim[0]*factor,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=dim[0]*factor,
            ),
            nn.BatchNorm2d(num_features=dim[0]*factor),
            self.act,
        )
        self.second_block = nn.Sequential(
            # Conv 1x1, linear act.
            ConvBlock(
                in_channels=dim[0]*factor,
                out_channels=dim[1],
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=dim[1]),
        )
        self.CE_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(in_features=dim[0]*factor, out_features=dim[2]),
            nn.ReLU(),
            nn.Linear(in_features=dim[2], out_features=dim[0]*factor),
            nn.Hardsigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connection is True and self.use_se is True:
            identity = x
            x = self.first_block(x)
            x += self.CE_block(x)

            return identity + self.second_block(x)

        elif self.use_res_connection is True and self.use_se is False:
            identity = x
            x = self.first_block(x)

            return identity + self.second_block(x)

        elif self.use_res_connection is False and self.use_se is True:
            x = self.first_block(x)
            x += self.CE_block(x)

            return self.second_block(x)

        elif self.use_res_connection is False and self.use_se is False:
            x = self.first_block(x)

            return self.second_block(x)


class Classifier(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)