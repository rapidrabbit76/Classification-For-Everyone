import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import (
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy

import datamodule
from model import MobileNetV2

sys.path.insert(1, os.path.abspath('..'))
import utils


class MobileNetV2Model(pl.LightningModule):

    def __init__(
            self,
            image_channels: int,
            n_classes: int,
            lr: int,
            alpha: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MobileNetV2(
            self.hparams.image_channels,
            self.hparams.n_classes,
            self.hparams.alpha,
        )
        self.model.initialize_weights()

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
        self.lof('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr
        )
        scheduler_dict = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True,
            ),
            'monitor': 'val_loss',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


def train():
    # Hyperparameters
    config = utils.get_config()
    hparams = config.hparams
    seed_everything(config.seed)

    # Dataloader
    if config.dataset == 'CIFAR10':
        dm = datamodule.CIFAR10DataModule(
            config.data_dir,
            image_size=config.image_size,
            batch_size=hparams.batch_size,
            rho=hparams.rho
        )
    else:
        dm = datamodule.CIFAR100DataModule(
            config.data_dir,
            image_size=config.image_size,
            batch_size=hparams.batch_size,
            rho=hparams.rho
        )

    # Model
    mobilenetv2 = MobileNetV2Model(
        image_channels=config.image_channels,
        n_classes=config.n_classes,
        lr=hparams.lr,
        alpha=hparams.alpha
    )

    # Logger
    wandb_logger = WandbLogger(
        name=f'{config.project_name}-{config.dataset}',
        project=config.project_name,
        save_dir=config.save_dir,
        log_model='all',
    )
    wandb_logger.experiment.config.update(hparams)
    wandb_logger.watch(mobilenetv2, log='all', log_freq=100)

    # Trainer setting
    callbacks = [
        EarlyStopping(
            monitor='val_acc',
            min_delta=0.00,
            patience=10,
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
    trainer.fit(mobilenetv2, datamodule=dm)
    trainer.test(mobilenetv2, datamodule=dm)

    # Finish
    wandb_logger.experiment.unwatch(mobilenetv2)

    # Model to Torchscript
    saved_model_path = utils.model_save(
        mobilenetv2,
        config.torchscript_model_save_path,
        config.project_name
    )

    # Save artifacts
    wandb_logger.experiment.save(saved_model_path)


if __name__=='__main__':
    train()
