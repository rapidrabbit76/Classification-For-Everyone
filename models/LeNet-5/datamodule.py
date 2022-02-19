from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST


class MnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        image_size: int = 32,
        batch_size: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.dataset = MNIST

    def prepare_data(self) -> None:
        """Dataset download"""
        self.dataset(self.hparams.data_dir, train=True, download=True)
        self.dataset(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            dataset = self.dataset(
                self.hparams.data_dir,
                train=True,
                transform=self.transform,
            )

            train_ds_size = int(len(dataset) * 0.8)
            val_ds_size = len(dataset) - train_ds_size

            self.train_ds, self.val_ds = random_split(
                dataset,
                [train_ds_size, val_ds_size],
            )

        if stage == "test" or stage is None:
            self.test_ds = self.dataset(
                self.hparams.data_dir,
                train=False,
                transform=self.transform,
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


class FMnistDataModule(MnistDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        image_size: int = 32,
        batch_size: int = 256,
    ):
        super().__init__(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size,
        )
        self.dataset = FashionMNIST


class EMnistDataModule(MnistDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        image_size: int = 32,
        batch_size: int = 256,
    ):
        super().__init__(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size,
        )
        self.dataset = EMNIST


class KMnistDataModule(MnistDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        image_size: int = 32,
        batch_size: int = 256,
    ):
        super().__init__(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size,
        )
        self.dataset = KMNIST


class QMnistDataModule(MnistDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        image_size: int = 32,
        batch_size: int = 256,
    ):
        super().__init__(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size,
        )
        self.dataset = QMNIST
