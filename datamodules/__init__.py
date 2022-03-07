from datamodules.MNIST import *
from datamodules.CIFAR import *
from datamodules.ImageNet import *

DATAMODULE_TABLE: Dict["str", pl.LightningDataModule] = {
    "MNIST": MnistDataModule,
    "FMNIST": FashionMnistDataModule,
    "EMNIST": EmnistDataModule,
    "KMNIST": KMnistDataModule,
    "CIFAR10": CIFAR10DataModule,
    "CIFAR100": CIFAR100DataModule,
    "ImageNet": ImageNetDataModule,
}

__all__ = [
    # MNIST
    "MnistDataModule",
    "FashionMnistDataModule",
    "EmnistDataModule",
    "KMnistDataModule",
    # CIFAR
    "CIFAR10DataModule",
    "CIFAR100DataModule",
    # ImageNet
    "ImageNetDataModule",
    # TABLE
    "DATAMODULE_TABLE",
]
