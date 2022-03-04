import torch
import torch.nn as nn

__all__ = ["ConvBlock", "BasicBlock", "BottleNeckBlock", "ResidualBlock", "Classifier"]


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
            layers.append(nn.ReLU())
        else:
            pass

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class BasicBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            is_first_layer: bool,
            is_first_block: bool,
    ):
        super().__init__()
        self.is_first_layer = is_first_layer
        self.is_first_block = is_first_block

        self.conv_3by3_1 = ConvBlock(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            act='ReLU',
        )
        self.conv_3by3_2 = ConvBlock(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            act='None',
        )

        self.conv_3by3_3 = ConvBlock(
            in_channels=dim//2,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=1,
            act='ReLU',
        )
        self.downsample = ConvBlock(
            in_channels=dim//2,
            out_channels=dim,
            kernel_size=1,
            stride=2,
            act='None',
        )

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.is_first_layer is False and self.is_first_block is True:
            x = self.conv_3by3_3(x)
            x = self.conv_3by3_2(x)

            identity = self.downsample(identity)

            x = x + identity

            return self.act(x)

        x = self.conv_3by3_1(x)
        x = self.conv_3by3_2(x)

        x = x + identity

        return self.act(x)


class BottleNeckBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            is_first_layer: bool,
            is_first_block: bool,
    ):
        super().__init__()
        self.is_first_layer = is_first_layer
        self.is_first_block = is_first_block

        # 두 상황 제외하고 공통으로 사용 가능한 Conv2D layer
        self.conv_1by1_1_common = ConvBlock(
            in_channels=dim*4,
            out_channels=dim,
            kernel_size=1,
            act='ReLU',
        )
        self.conv_3by3_common = ConvBlock(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            act='ReLU',
        )
        self.conv_1by1_2_common = ConvBlock(
            in_channels=dim,
            out_channels=dim*4,
            kernel_size=1,
            act='None',
        )

        # Conv2_1, dim=64 -> 첫 conv layer & 첫 conv block
        self.conv2_1_1by1_1 = ConvBlock(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            act='ReLU',
        )
        self.conv2_1_3by3 = ConvBlock(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            act='ReLU',
        )
        self.conv2_1_1by1_2 = ConvBlock(
            in_channels=dim,
            out_channels=dim*4,
            kernel_size=1,
            act='None',
        )
        self.scale = ConvBlock(
            in_channels=dim,
            out_channels=dim*4,
            kernel_size=1,
            act='None',
        )

        # Conv3_1, dim=128 -> 이외 conv layer & 첫 conv block
        self.conv3_1_1by1_1 = ConvBlock(
            in_channels=dim*2,
            out_channels=dim,
            kernel_size=1,
            stride=2,
            act='ReLU',
        )
        self.conv3_1_3by3 = ConvBlock(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            act='ReLU',
        )
        self.conv3_1_1by1_2 = ConvBlock(
            in_channels=dim,
            out_channels=dim*4,
            kernel_size=1,
            act='None',
        )
        self.downsample = ConvBlock(
            in_channels=dim*2,
            out_channels=dim*4,
            kernel_size=1,
            stride=2,
            act='None',
        )

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.is_first_layer is True and self.is_first_block is True:
            x = self.conv2_1_1by1_1(x)
            x = self.conv2_1_3by3(x)
            x = self.conv2_1_1by1_2(x)

            identity = self.scale(identity)

            x = x + identity

            return self.act(x)

        elif self.is_first_layer is False and self.is_first_block is True:
            x = self.conv3_1_1by1_1(x)
            x = self.conv3_1_3by3(x)
            x = self.conv3_1_1by1_2(x)

            identity = self.downsample(identity)

            x = x + identity

            return self.act(x)

        x = self.conv_1by1_1_common(x)
        x = self.conv_3by3_common(x)
        x = self.conv_1by1_2_common(x)

        x = x + identity

        return self.act(x)


class ResidualBlock(nn.Module):

    def __init__(
            self,
            num_layers: int,
            idx_layer: int,
            nblocks: int,
            dim: int,
    ):
        super().__init__()
        layers = []

        if num_layers < 50:
            for idx_block in range(nblocks):
                if idx_layer == 0:
                    layers += [BasicBlock(dim, True, False)]
                elif idx_layer != 0 and idx_block == 0:
                    layers += [BasicBlock(dim, False, True)]
                else:
                    layers += [BasicBlock(dim, False, False)]
        else:
            for idx_block in range(nblocks):
                if idx_layer == 0 and idx_block == 0:
                    layers += [BottleNeckBlock(dim, True, True)]
                elif idx_layer != 0 and idx_block == 0:
                    layers += [BottleNeckBlock(dim, False, True)]
                else:
                    layers += [BottleNeckBlock(dim, False, False)]

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Classifier(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout_rate: float,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)