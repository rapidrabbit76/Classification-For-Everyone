from math import log
import torch
from torch import nn

from .blocks import Classifier, ConvBlock


class VGG11(nn.Module):
    def __init__(
        self,
        n_classes: int = 1000,
        use_bn: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # 1 conv
            ConvBlock(3, 64, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 2 conv
            ConvBlock(64, 128, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 3 conv
            ConvBlock(128, 256, norm=use_bn),
            ConvBlock(256, 256, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 4 conv
            ConvBlock(256, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 5 conv
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.classfier = nn.Sequential(
            nn.Flatten(),
            Classifier((512 * (image_size // 2 ** 5) ** 2), 4096, n_classes, 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return self.classfier(x)


class VGG13(VGG11):
    def __init__(
        self,
        n_classes: int = 1000,
        use_bn: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__(n_classes, use_bn, image_size)

        self.feature_extractor = nn.Sequential(
            # 1 conv
            ConvBlock(3, 64, norm=use_bn),
            ConvBlock(64, 64, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 2 conv
            ConvBlock(64, 128, norm=use_bn),
            ConvBlock(128, 128, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 3 conv
            ConvBlock(128, 256, norm=use_bn),
            ConvBlock(256, 256, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 4 conv
            ConvBlock(256, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 5 conv
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )


class VGG16_C(VGG11):
    def __init__(
        self,
        n_classes: int = 1000,
        use_bn: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__(n_classes, use_bn, image_size)

        self.feature_extractor = nn.Sequential(
            # 1 conv
            ConvBlock(3, 64, norm=use_bn),
            ConvBlock(64, 64, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 2 conv
            ConvBlock(64, 128, norm=use_bn),
            ConvBlock(128, 128, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 3 conv
            ConvBlock(128, 256, norm=use_bn),
            ConvBlock(256, 256, norm=use_bn),
            ConvBlock(256, 256, kernel_size=1, padding=0, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 4 conv
            ConvBlock(256, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, kernel_size=1, padding=0, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 5 conv
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, kernel_size=1, padding=0, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )


class VGG16_D(VGG11):
    def __init__(
        self,
        n_classes: int = 1000,
        use_bn: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__(n_classes, use_bn, image_size)

        self.feature_extractor = nn.Sequential(
            # 1 conv
            ConvBlock(3, 64, norm=use_bn),
            ConvBlock(64, 64, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 2 conv
            ConvBlock(64, 128, norm=use_bn),
            ConvBlock(128, 128, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 3 conv
            ConvBlock(128, 256, norm=use_bn),
            ConvBlock(256, 256, norm=use_bn),
            ConvBlock(256, 256, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 4 conv
            ConvBlock(256, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 5 conv
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )


class VGG19(VGG11):
    def __init__(
        self,
        n_classes: int = 1000,
        use_bn: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__(n_classes, use_bn, image_size)

        self.feature_extractor = nn.Sequential(
            # 1 conv
            ConvBlock(3, 64, norm=use_bn),
            ConvBlock(64, 64, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 2 conv
            ConvBlock(64, 128, norm=use_bn),
            ConvBlock(128, 128, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 3 conv
            ConvBlock(128, 256, norm=use_bn),
            ConvBlock(256, 256, norm=use_bn),
            ConvBlock(256, 256, norm=use_bn),
            ConvBlock(256, 256, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 4 conv
            ConvBlock(256, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 5 conv
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            ConvBlock(512, 512, norm=use_bn),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
