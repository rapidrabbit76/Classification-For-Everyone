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
)
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torchmetrics import Accuracy

import datamodule
from model import Inception_v3, MODEL_RETURN_TYPE

from typing import Tuple, Dict

sys.path.insert(1, os.path.abspath(".."))
import utils

_batch_type = Tuple[torch.Tensor, torch.Tensor]
_step_return_type = Tuple[Dict[str, torch.Tensor], torch.Tensor]


class InceptionV3Module(pl.LightningModule):

    def __init__(
        self,
        image_channels: int,
        norm: bool,
        n_classes: int,
        lr: float,
        lr_exponential_rate: float,
        label_smoothing: float,
        weight_decay: float,
        eps: float,
        gradient_clipping: float,
        loss_weight: float,
        aux_loss_weight: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Inception_v3(
            image_channels=self.hparams.image_channels,
            num_classes=self.hparams.n_classes,
            norm=self.hparams.norm,
        )
        self.model.initialize_weights()

        # Metrics
        self.loss = nn.CrossEntropyLoss(label_smoothing=0)
        self.accuracy = Accuracy()

    def forward(self, x: torch.Tensor) -> MODEL_RETURN_TYPE:
        return self.model(x)

    def step(self, batch: _batch_type) -> _step_return_type:
        x, y = batch
        logits = self(x)

        # Inception aux phase branch
        if isinstance(logits, list):
            logits, aux_logits = logits
            loss = self.loss(logits, y)
            aux_loss = self.loss(aux_logits, y)
        else:
            loss = self.loss(logits, y)
            aux_loss = 0

        pred = torch.argmax(logits, dim=1)
        acc = self.accuracy(pred, y)

        return {
            'loss': loss,
            'aux_loss': aux_loss,
        }, acc

    def training_step(
        self,
        batch: _batch_type,
        batch_idx: int,
    ) -> torch.Tensor:
        loss_dict, acc = self.step(batch)

        loss = loss_dict['loss'] * self.hparams.loss_weight + loss_dict[
            'aux_loss'] * self.hparams.aux_loss_weight

        self.log("main_loss", loss_dict['loss'], prog_bar=True)
        self.log("aux_loss", loss_dict['aux_loss'], prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: _batch_type,
        batch_idx: int,
    ) -> torch.Tensor:
        loss_dict, acc = self.step(batch)
        loss = loss_dict['loss']
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(
        self,
        batch: _batch_type,
        batch_idx: int,
    ) -> torch.Tensor:
        loss_dict, acc = self.step(batch)
        loss = loss_dict['loss']
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.RMSprop(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps,
        )

        lr_scheduler_config = {
            "scheduler":
            optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=self.hparams.lr_exponential_rate,
            ),
            "interval":
            "step",
            'frequency':
            2,
            "monitor":
            "val_loss",
            "strict":
            True,
        }
        return [optimizer], [lr_scheduler_config]


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
    model = InceptionV3Module(
        image_channels=config.image_channels,
        norm=hparams.norm,
        n_classes=config.n_classes,
        lr=hparams.lr,
        lr_exponential_rate=hparams.lr_exponential_rate,
        label_smoothing=hparams.label_smoothing,
        weight_decay=hparams.weight_decay,
        eps=hparams.eps,
        gradient_clipping=hparams.gradient_clipping,
        loss_weight=hparams.loss_weight,
        aux_loss_weight=hparams.aux_loss_weight,
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
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode="max",
        ),
    ]

    trainer: pl.Trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=hparams.epochs,
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
        hparams.model_type,
    )

    # Save artifacts
    wandb_logger.experiment.save(saved_model_path)


if __name__ == "__main__":
    train()
