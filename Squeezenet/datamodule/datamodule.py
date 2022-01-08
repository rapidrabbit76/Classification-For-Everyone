from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .dataset import CIFAR100Dataset, CIFAR10Dataset


class CIFAR100DataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "./data",
        image_size: int = 224,
        batch_size: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """Dataset download"""
        CIFAR100Dataset(self.hparams.data_dir, True, download=True)
        CIFAR100Dataset(self.hparams.data_dir, False, download=True)
        self.dataset = CIFAR100Dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_ds = self.dataset(self.hparams.data_dir, True)
            train_ds_size = int(len(self.train_ds) * 0.8)
            val_ds_size = len(self.train_ds) - train_ds_size
            self.train_ds, self.val_ds = random_split(
                self.train_ds,
                [train_ds_size, val_ds_size],
            )

        if stage == "test" or stage is None:
            self.test_ds = self.dataset(
                self.hparams.data_dir,
                "val",
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=8,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=8,
        )


class CIFAR10DataModule(CIFAR100DataModule):

    def prepare_data(self) -> None:
        CIFAR10Dataset(self.hparams.data_dir, True, download=True)
        CIFAR10Dataset(self.hparams.data_dir, False, download=True)
        self.dataset = CIFAR10Dataset