from typing import *

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100

__all__ = ["CIFAR10DataModule", "CIFAR100DataModule"]


class CIFARDataModuleBase(pl.LightningDataModule):
    def __init__(
        self,
        DATASET: Union[CIFAR10, CIFAR100],
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

    def prepare_data(self) -> None:
        """Dataset download"""
        self.Dataset(self.hparams.root_dir, train=True, download=True)
        self.Dataset(self.hparams.root_dir, train=False, download=True)

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


def CIFAR10DataModule(**kwargs):
    return CIFARDataModuleBase(CIFAR10, **kwargs)


def CIFAR100DataModule(**kwargs):
    return CIFARDataModuleBase(CIFAR100, **kwargs)
