from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from .dataset import CIFAR10DataSet


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: str = './data',
            image_size: int = 224,
            batch_size: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_transforms = A.Compose([
            A.Resize(256, 256),
            A.RandomResizedCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=(0.49139968, 0.48215841, 0.44653091),
                std=(0.24703223, 0.24348513, 0.26158784),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensor(),
        ], )

        self.test_transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.49139968, 0.48215841, 0.44653091),
                std=(0.24703223, 0.24348513, 0.26158784),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensor(),
        ], )

    def prepare_data(self) -> None:
        """Dataset download"""
        CIFAR10DataSet(
            self.hparams.data_dir,
            train=True,
            transform=transform_lib.ToTensor(),
            download=True
        )
        CIFAR10DataSet(
            self.hparams.data_dir,
            train=False,
            transform=transform_lib.ToTensor(),
            download=True
        )

    def train_dataloader(self) -> DataLoader:

        dataset = CIFAR10DataSet(
            self.hparams.data_dir,
            train=True,
            download=False,
            transform=self.train_transforms
        )
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_ds, _ = random_split(
            dataset,
            [train_size, val_size],
        )
        return DataLoader(
            train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:

        dataset = CIFAR10DataSet(
            self.hparams.data_dir,
            train=True,
            download=False,
            transform=self.test_transforms
        )
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        _, val_ds = random_split(
            dataset,
            [train_size, val_size]
        )
        return DataLoader(
            val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:

        test_ds = CIFAR10DataSet(
            self.hparams.data_dir,
            train=False,
            download=False,
            transform=self.test_transforms
        )
        return DataLoader(
            test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
        )