from typing import *

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import cv2
import numpy as np


class ImageNet(ImageFolder):
    def __init__(
        self,
        root_dir: str,
        transforms: Callable = None,
        **kargs,
    ) -> None:
        self.transforms = transforms
        super().__init__(root_dir, transforms, loader=self._image_loader)

    def _image_loader(cls, path: str) -> np.ndarray:
        image = cv2.imread(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(index)

        if self.transform != None:
            image = self.transforms(image)
        return image, target


class ImageNetDataModuleBase(pl.LightningDataModule):
    def __init__(
        self,
        DATASET: ImageNet,
        root_dir: str,
        train_transforms: Callable,
        val_transforms: Callable,
        test_transforms: Callable,
        batch_size: int = 256,
        num_workers: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "root_dir": root_dir,
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
        )
        self.Dataset = DATASET
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # split dataset to train, val
            ds = self.Dataset(self.hparams.root_dir, train=True)
            targets = ds.targets
            train_idx, val_idx = train_test_split(
                np.arange(len(targets)),
                test_size=0.2,
                shuffle=True,
                stratify=targets,
            )

            # build dataset, different transforms
            self.train_ds = Subset(
                self.Dataset(
                    self.hparams.root_dir,
                    train=True,
                    transform=self.train_transforms,
                ),
                train_idx,
            )
            self.val_ds = Subset(
                self.Dataset(
                    self.hparams.root_dir,
                    train=True,
                    transform=self.test_transforms,
                ),
                val_idx,
            )

        if stage == "test" or stage is None:
            self.test_ds = self.Dataset(
                self.hparams.root_dir,
                train=False,
                transform=self.test_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


def ImageNetDataModule(**kwargs):
    return ImageNetDataModuleBase(ImageNet, **kwargs)
