from asyncio.constants import ACCEPT_RETRY_DELAY
from ctypes import Union
from curses import meta
import os
from turtle import down
from typing import Any, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image
from torch.utils.data import Dataset
from torchaudio import datasets
from torchvision import transforms
from torchvision.datasets import CIFAR100
import torchvision.datasets as datasets
import pandas as pd
import numpy as np


transform = A.Compose(
    [
        A.Resize(256, 256),
        # A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensor(),
    ],
)


class CIFAR100Dataset(CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__(
            root,
            train=train,
            download=download,
        )

        self.transform = A.Compose(
            [
                A.Resize(256, 256),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensor(),
            ],
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = self.data[index], self.targets[index]
        # image = Image.open(image_path).convert("RGV")
        image = self.transform(image=image)["image"]
        return image, target


if __name__ == "__main__":
    root_dir = "../data"
    train_ds = CIFAR100Dataset(root_dir, train=True, download=True)
    test_ds = CIFAR100Dataset(root_dir, train=False, download=True)

    for image, target in train_ds:
        print(image.shape, target)
        break

    for image, target in test_ds:
        print(image.shape, target)
        break
