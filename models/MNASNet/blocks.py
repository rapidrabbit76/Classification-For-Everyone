import torch
import torch.nn as nn
from typing import List

__all__ = ["ConvBlock", "SepConvBlock", "SEBlock", "MBConvBlock", "Classifier"]


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            act: str = 'ReLU',
            bias: bool = False,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(num_features=out_channels)
        ]

        if act == 'ReLU':
            layers.append(nn.ReLU(inplace=True))

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class SepConvBlock(nn.Module):

    def __init__(
            self,
            dim: List[int],
            factor: float,
            stride: int,
            padding: int
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(
                in_channels=expand_width(dim[0], factor),
                out_channels=expand_width(dim[0], factor),
                kernel_size=3,
                stride=stride,
                padding=padding,
                groups=dim[0],
            ),
            ConvBlock(
                in_channels=expand_width(dim[0], factor),
                out_channels=expand_width(dim[1], factor),
                kernel_size=1,
                act='None'
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SEBlock(nn.Module):

    def __init__(
            self,
            dim: List[int],
            factor: float,
            reduction_ratio: int = 4
    ) -> None:
        super().__init__()

        self.CE_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=expand_width(dim[0], factor),
                out_channels=expand_width(
                    dim=dim[0],
                    factor=factor,
                    ratio=reduction_ratio,
                    use_ratio=True
                ),
                kernel_size=1
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=expand_width(
                    dim=dim[0],
                    factor=factor,
                    ratio=reduction_ratio,
                    use_ratio=True
                ),
                out_channels=expand_width(dim[0], factor),
                kernel_size=1
            ),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self.CE_block(x))


class MBConvBlock(nn.Module):

    def __init__(
            self,
            dim: List[int],
            factor: float,
            kernel: int,
            padding: int,
            stride: int,
            reduction_ratio: int = 4,
            use_se: bool = False,
    ) -> None:
        super().__init__()
        self.use_se = use_se
        self.use_res_connection = (stride == 1) and (dim[0] == dim[1])

        self.first_block = nn.Sequential(
            # Conv 1x1
            ConvBlock(
                in_channels=dim[0],
                out_channels=expand_width(dim[0], factor),
                kernel_size=1,
            ),
            # Dwise kernel x kennel
            ConvBlock(
                in_channels=expand_width(dim[0], factor),
                out_channels=expand_width(dim[0], factor),
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=expand_width(dim[0], factor),
            )
        )

        self.second_block = nn.Sequential(
            # Conv 1x1, linear act.
            ConvBlock(
                in_channels=expand_width(dim[0], factor),
                out_channels=dim[1],
                kernel_size=1,
                act='None'
            )
        )

        self.SEBlock = SEBlock(
            dim=dim,
            factor=factor,
            reduction_ratio=reduction_ratio
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.use_res_connection is True and self.use_se is True:
            identity = x
            x = self.first_block(x)
            x = self.SEBlock(x)

            return identity + self.second_block(x)

        elif self.use_res_connection is True and self.use_se is False:
            identity = x
            x = self.first_block(x)

            return identity + self.second_block(x)

        elif self.use_res_connection is False and self.use_se is True:
            x = self.first_block(x)
            x = self.SEBlock(x)

            return self.second_block(x)

        else:
            x = self.first_block(x)

            return self.second_block(x)


def expand_width(
        dim: int,
        factor: float,
        ratio: int = 4,
        use_ratio: bool = False,
) -> int:
    if use_ratio is True:
        return int(dim//ratio)
    else:
        return int(dim*factor)


class Classifier(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout_rate: float,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)