import torch
import torch.nn as nn
from typing import List

__all__ = ["ConvBlock", "SEBlock", "MBConv", "MBConvBlock", "Classifier"]


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            act: str = 'SiLU',
            use_bn: bool = True,
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
            )
        ]

        if use_bn is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if act == 'SiLU':
            layers.append(nn.SiLU())
        elif act == 'Sigmoid':
            layers.append(nn.Sigmoid())

        self.Conv2d = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Conv2d(x)


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
            nn.SiLU(),
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
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self.CE_block(x))


class MBConv(nn.Module):

    def __init__(
            self,
            dim: List[int],
            factor: float,
            kernel_size: int,
            padding: int,
            stride: int,
            reduction_ratio: int = 4
    ) -> None:
        super().__init__()
        self.use_res_connection = (stride == 1) and (dim[0] == dim[1])

        self.MBConv = nn.Sequential(
            # 1x1 Conv
            ConvBlock(
                in_channels=dim[0],
                out_channels=expand_width(dim[0], factor),
                kernel_size=1
            ),
            # DWConv
            ConvBlock(
                in_channels=expand_width(dim[0], factor),
                out_channels=expand_width(dim[0], factor),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=expand_width(dim[0], factor)
            ),
            # SEBlock
            SEBlock(
                dim=dim,
                factor=factor,
                reduction_ratio=reduction_ratio
            ),
            # 1x1 Conv
            ConvBlock(
                in_channels=expand_width(dim[0], factor),
                out_channels=dim[1],
                kernel_size=1,
                act='None'
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.use_res_connection is True:
            identity = x

            return identity + self.MBConv(x)

        else:
            return self.MBConv(x)


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


class MBConvBlock(nn.Module):

    def __init__(
            self,
            numLayer: int,
            dim: List[int],
            factor: float,
            kernel_size: int,
            padding: int,
            stride: int,
            reduction_ratio: int = 4,
    ) -> None:
        super().__init__()
        layers = []

        for idx in range(numLayer):
            if idx > 0:
                stride = 1
                dim[0] = dim[1]

            layers.append(
                MBConv(
                    dim=dim,
                    factor=factor,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    reduction_ratio=reduction_ratio
                )
            )

        self.MBConvBlock = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.MBConvBlock(x)


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