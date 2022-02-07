from ast import parse
import os
import sys
import argparse

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import (
    EarlyStopping,
    TQDMProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torchmetrics import Accuracy

import datamodule
from model import ResNext

from typing import Tuple, Dict
import torchvision
import torchsummary

sys.path.insert(1, os.path.abspath(".."))
import utils

_batch_type = Tuple[torch.Tensor, torch.Tensor]
_step_return_type = Tuple[torch.Tensor, torch.Tensor]


class ResNeXtModule(pl.LightningModule):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.save_hyperparameters(config)

        self.model = ResNext(
            model_type=self.hparams.model_type,
            image_channels=self.hparams.image_channels,
            num_classes=self.hparams.num_classes,
            norm=self.hparams.norm,
            groups=self.hparams.groups,
            depth=self.hparams.depth,
            basewidth=self.hparams.basewidth,
        )

        # Metrics
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self, batch: _batch_type) -> _step_return_type:
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        pred = torch.argmax(logits, dim=1)
        acc = self.accuracy(pred, y)
        return loss, acc

    def training_step(
        self,
        batch: _batch_type,
        batch_idx: int,
    ) -> torch.Tensor:
        loss, acc = self.step(batch)

        self.log_dict({
            'train_loss': loss,
            'train_acc': acc,
        })

        return loss

    def validation_step(
        self,
        batch: _batch_type,
        batch_idx: int,
    ) -> torch.Tensor:
        loss, acc = self.step(batch)
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc,
        })

        return loss

    def test_step(
        self,
        batch: _batch_type,
        batch_idx: int,
    ) -> torch.Tensor:
        loss, acc = self.step(batch)
        self.log_dict({
            'test_loss': loss,
            'test_acc': acc,
        })
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.hparams.lr_scheduler_gamma,
        )

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': "epoch",
            'frequency': self.hparams.epochs // 3
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_dict,
        }


def train():
    # Hyperparameters
    config = utils.get_config()
    seed_everything(config.seed)

    # Dataloader
    if config.dataset == 'CIFAR10':
        dm = datamodule.CIFAR10DataModule(
            config.data_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
        )
    else:
        dm = datamodule.CIFAR100DataModule(
            config.data_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
        )

    # Model
    model = ResNeXtModule(config)

    # Logger
    wandb_logger = WandbLogger(
        name=f"{config.project_name}-{config.dataset}",
        project=config.project_name,
        save_dir=config.save_dir,
        log_model="all",
    )

    wandb_logger.watch(model, log="all", log_freq=100)

    # Trainer setting
    callbacks = [
        TQDMProgressBar(refresh_rate=5),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    trainer: pl.Trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        precision=config.precision,
        max_epochs=config.epochs,
        callbacks=callbacks,
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
    )

    # Save artifacts
    wandb_logger.experiment.save(saved_model_path)


if __name__ == "__main__":
    train()