from datamodules.MNIST import *
from datamodules.CIFAR import *

DATAMODULE_TABLE: Dict["str", pl.LightningDataModule] = {
    "MNIST": MnistDataModule,
    "FMNIST": FashionMnistDataModule,
    "EMNIST": EmnistDataModule,
    "KMNIST": KMnistDataModule,
    "CIFAR10": CIFAR10DataModule,
    "CIFAR100": CIFAR100DataModule,
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
    # TABLE
    "DATAMODULE_TABLE",
]
