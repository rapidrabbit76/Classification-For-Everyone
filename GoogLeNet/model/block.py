import torch
import torch.nn as nn

# module 오탈자 수정, padding str -> int로 수정
# padding 수정 필요
class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
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
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-3)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

# 상속 nn.Module로 교체
class InceptionBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels_1by1: int,
            out_channels_3by3_reduce: int,
            out_channels_3by3: int,
            out_channels_5by5_reduce: int,
            out_channels_5by5: int,
            out_channels_pool_proj: int,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.out_channels_1by1 = out_channels_1by1
        self.out_channels_3by3_reduce = out_channels_3by3_reduce
        self.out_channels_3by3 = out_channels_3by3
        self.out_channels_5by5_reduce = out_channels_5by5_reduce
        self.out_channels_5by5 = out_channels_5by5
        self.out_channels_pool_proj = out_channels_pool_proj

        self.maxpooling2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = ConvBlock(self.in_channels, self.out_channels_1by1, kernel_size=1)(x)

        x2_1 = ConvBlock(self.in_channels, self.out_channels_3by3_reduce, kernel_size=1)(x)
        x2_2 = ConvBlock(self.out_channels_3by3_reduce, self.out_channels_3by3, kernel_size=3)(x2_1)

        x3_1 = ConvBlock(self.in_channels, self.out_channels_5by5_reduce, kernel_size=1)(x)
        x3_2 = ConvBlock(self.out_channels_5by5_reduce, self.out_channels_5by5, kernel_size=5)(x3_1)

        x4_1 = self.maxpooling2d(x)
        x4_2 = ConvBlock(self.in_channels, self.out_channels_pool_proj, kernel_size=1)(x4_1)

        return torch.cat([x1, x2_2, x3_2, x4_2], dim=1)


class AuxClassifier(nn.Module):

    def __init__(
            self,
            in_features: int,
            dim: int,
            out_features: int,
            dropout_rate: float = 0.4,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class Classifier(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout_rate: float = 0.4,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)