import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pytorch_lightning.callbacks import (
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torchmetrics import Accuracy

import datamodule
from model import AlexNet

sys.path.insert(1, os.path.abspath('..'))
import utils


class AlexNetModel(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            n_classes: int,
            lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AlexNet(
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
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.step(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.step(batch)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler_dict = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=7,
            ),
            'interval': 'step',
            'monitor': 'train_loss',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


def train():
    # Hyperparameters
    config = utils.get_config()
    hparams = config.hparams
    seed_everything(config.seed)

    # Dataloader
    dm = datamodule.CIFAR10DataModule(config.data_dir)
    n_classes = 10

    # Model
    alexnet = AlexNetModel(
        config.image_channels,
        n_classes,
        hparams.lr,
    )

    # Logger
    wandb_logger = WandbLogger(
        name=f'{config.project_name}-{config.dataset}',
        project=config.project_name,
        save_dir=config.save_dir,
        log_model='all',
    )
    wandb_logger.experiment.config.update(hparams)
    wandb_logger.watch(alexnet, log='all', log_freq=100)

    # Trainer setting
    callbacks = [
        EarlyStopping(
            monitor='val_acc',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='max',
        ),
        TQDMProgressBar(refresh_rate=10),
    ]

    trainer: pl.Trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=hparams.epochs,
        callbacks=callbacks,
    )

    # Train
    trainer.fit(alexnet, datamodule=dm)
    trainer.test(alexnet, datamodule=dm)

    # Finish
    wandb_logger.experiment.unwatch(alexnet)

    # Model to Torchscript
    script = alexnet.to_torchscript()
    torch.jit.save(script, config.torchscript_model_save_path)
    wandb_logger.experiment.save(config.torchscript_model_save_path)


if __name__ == '__main__':
    train()