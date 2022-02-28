from datamodules.MNIST import *
from datamodules.CIFAR import *


__all__ = [
    # MNIST
    "MnistDataModule",
    "FashionMnistDataModule",
    "EmnistDataModule",
    # CIFAR
    "CIFAR10DataModule",
    "CIFAR100DataModule",
]
