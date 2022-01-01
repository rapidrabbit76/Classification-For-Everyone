import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import (
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torchmetrics import Accuracy

import datamodule
from model import LeNet5

sys.path.insert(1, os.path.abspath(".."))
import utils


class LeNet5Model(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = LeNet5(
            self.hparams.in_channels,
            self.hparams.n_classes,
        )

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self, batch):
        x, y = batch
        logit = self(x)
        loss = self.loss(logit, y)
        preds = torch.argmax(logit, dim=1)
        acc = self.accuracy(preds, y)
        return loss, acc

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.step(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


def train():
    # Hyperparameters
    config = utils.get_config()
    hparams = config.hparams
    seed_everything(config.seed)

    # Dataloader
    dm = datamodule.QMnistDataModule(config.data_dir)
    n_classes = 10

    # Model
    lenet5 = LeNet5Model(
        config.image_channals,
        n_classes,
        hparams.lr,
    )

    # Logger
    wandb_logger = WandbLogger(
        name=f"{config.project_name}-{config.dataset}",
        project=config.project_name,
        save_dir=config.save_dir,
        log_model="all",
    )
    wandb_logger.experiment.config.update(hparams)
    wandb_logger.watch(lenet5, log="all", log_freq=100)

    callbacks = [
        EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode="max",
        ),
        TQDMProgressBar(refresh_rate=10),
    ]

    # Trainer
    trainer: pl.Trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=hparams.epochs,
        callbacks=callbacks,
    )

    # Train
    trainer.fit(lenet5, datamodule=dm)
    trainer.test(lenet5, datamodule=dm)

    # Finish
    wandb_logger.experiment.unwatch(lenet5)

    # Model to Torchscript
    script = lenet5.to_torchscript()
    torch.jit.save(script, config.torchscript_model_save_path)
    wandb_logger.experiment.save(config.torchscript_model_save_path)


if __name__ == "__main__":
    train()
