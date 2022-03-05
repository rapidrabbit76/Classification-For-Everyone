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
from model import EfficientNet

sys.path.insert(1, os.path.abspath('..'))
import utils


class EfficientNetModel(pl.LightningModule):

    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.save_hyperparameters()
        model_size = self.hparams.config.model_size

        self.model = EfficientNet(
            image_channels=self.hparams.config.image_channels,
            n_classes=self.hparams.config.n_classes,
            width_coefficient=self.hparams.config.model_type[model_size][0],
            depth_coefficient=self.hparams.config.model_type[model_size][1],
            dropout_rate=self.hparams.config.dropout_rate
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
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.config.lr,
            weight_decay=self.hparams.config.weight_decay,
        )
        scheduler_dict = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.config.scheduler_factor,
                patience=self.hparams.config.scheduler_patience,
                verbose=True,
            ),
            'monitor': 'val_loss',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


def train():
    # Hyperparameters
    config = utils.get_config()
    seed_everything(config.seed)

    # Dataloader
    if config.dataset == 'CIFAR10':
        dm = datamodule.CIFAR10DataModule(
            config.data_dir,
            image_size=config.model_type[config.model_size][2],
            batch_size=config.batch_size,
        )
    else:
        dm = datamodule.CIFAR100DataModule(
            config.data_dir,
            image_size=config.model_type[config.model_size][2],
            batch_size=config.batch_size,
        )

    # Model
    efficientnet = EfficientNetModel(config=config)

    # Logger
    wandb_logger = WandbLogger(
        name=f'{config.project_name}-{config.dataset}',
        project=config.project_name,
        save_dir=config.save_dir,
        log_model='all',
    )

    wandb_logger.watch(efficientnet, log='all', log_freq=100)

    # Trainer setting
    callbacks = [
        EarlyStopping(
            monitor='val_acc',
            min_delta=0.00,
            patience=config.earlystopping_patience,
            verbose=True,
            mode='max',
        ),
        TQDMProgressBar(refresh_rate=10),
    ]

    trainer: pl.Trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=config.epochs,
        callbacks=callbacks,
    )

    # Train
    trainer.fit(efficientnet, datamodule=dm)
    trainer.test(efficientnet, datamodule=dm)

    # Finish
    wandb_logger.experiment.unwatch(efficientnet)

    # Model to Torchscript
    saved_model_path = utils.model_save(
        efficientnet,
        config.torchscript_model_save_path,
        config.project_name+config.model_size
    )

    # Save artifacts
    wandb_logger.experiment.save(saved_model_path)


if __name__=='__main__':
    train()
