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
from model import SqueezeNet

sys.path.insert(1, os.path.abspath(".."))
import utils


class SqueezeNetModule(pl.LightningModule):

    def __init__(
        self,
        image_channals: int,
        n_classes: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SqueezeNet(
            self.hparams.image_channals,
            self.hparams.n_classes,
        )
        self.model.initialize_weights()

        # Metrics
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
        return optim.SGD(self.model.parameters(),
                         lr=self.hparams.lr,
                         momentum=0.9,
                         weight_decay=0.0002)


def train():
    # Hyperparameters
    config = utils.get_config()
    hparams = config.hparams
    seed_everything(config.seed)

    # Dataloader
    if config.dataset == 'CIFAR10':
        dm = datamodule.CIFAR10DataModule(
            config.data_dir,
            batch_size=hparams.batch_size,
        )
    else:
        dm = datamodule.CIFAR100DataModule(
            config.data_dir,
            batch_size=hparams.batch_size,
        )

    # Model
    model = SqueezeNetModule(
        config.image_channals,
        config.n_classes,
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
    wandb_logger.watch(model, log="all", log_freq=100)

    # Trainer setting

    trainer: pl.Trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=hparams.epochs,
    )

    # Train
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    # Finish
    wandb_logger.experiment.unwatch(model)

    # Model to Torchscript
    saved_model_path = utils.model_save(
        model,
        config.torchscript_model_save_path,
        config.model_name + '.pt',
    )

    # Save artifacts
    wandb_logger.experiment.save(saved_model_path)


if __name__ == "__main__":
    train()
