import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)

        return x


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

        self.conv_3by3_1 = ConvBlock(dim, dim, kernel_size=3, padding=1)
        self.conv_3by3_2 = ConvBlock(dim, dim, kernel_size=3, padding=1)

        self.conv_3by3_3 = ConvBlock(dim//2, dim, kernel_size=3, stride=2, padding=1)
        self.downsample = ConvBlock(dim//2, dim, kernel_size=1, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.is_first_layer is False and self.is_first_block is True:
            x = self.conv_3by3_3(x)
            x = self.relu(x)
            x = self.conv_3by3_2(x)

            identity = self.downsample(identity)
            identity = self.bn(identity)

            x = x + identity

            return self.relu(x)

        x = self.conv_3by3_1(x)
        x = self.relu(x)
        x = self.conv_3by3_2(x)

        x = x + identity

        return self.relu(x)


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
        self.conv_1by1_1_common = ConvBlock(dim*4, dim, kernel_size=1)
        self.conv_3by3_common = ConvBlock(dim, dim, kernel_size=3, padding=1)
        self.conv_1by1_2_common = ConvBlock(dim, dim*4, kernel_size=1)

        # Conv2_1, dim=64 -> 첫 conv layer & 첫 conv block
        self.conv2_1_1by1_1 = ConvBlock(dim, dim, kernel_size=1)
        self.conv2_1_3by3 = ConvBlock(dim, dim, kernel_size=3, padding=1)
        self.conv2_1_1by1_2 = ConvBlock(dim, dim*4, kernel_size=1)
        self.scale = ConvBlock(dim, dim*4, kernel_size=1)

        # Conv3_1, dim=128 -> 이외 conv layer & 첫 conv block
        self.conv3_1_1by1_1 = ConvBlock(dim*2, dim, kernel_size=1, stride=2)
        self.conv3_1_3by3 = ConvBlock(dim, dim, kernel_size=3, padding=1)
        self.conv3_1_1by1_2 = ConvBlock(dim, dim*4, kernel_size=1)
        self.downsample = ConvBlock(dim*2, dim*4, kernel_size=1, stride=2)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=dim*4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.is_first_layer is True and self.is_first_block is True:
            x = self.conv2_1_1by1_1(x)
            x = self.relu(x)
            x = self.conv2_1_3by3(x)
            x = self.relu(x)
            x = self.conv2_1_1by1_2(x)

            identity = self.scale(identity)
            identity = self.bn(identity)

            x = x + identity

            return self.relu(x)

        elif self.is_first_layer is False and self.is_first_block is True:
            x = self.conv3_1_1by1_1(x)
            x = self.relu(x)
            x = self.conv3_1_3by3(x)
            x = self.relu(x)
            x = self.conv3_1_1by1_2(x)

            identity = self.downsample(identity)
            identity = self.bn(identity)

            x = x + identity

            return self.relu(x)

        x = self.conv_1by1_1_common(x)
        x = self.relu(x)
        x = self.conv_3by3_common(x)
        x = self.relu(x)
        x = self.conv_1by1_2_common(x)

        x = x + identity

        return self.relu(x)


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
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)