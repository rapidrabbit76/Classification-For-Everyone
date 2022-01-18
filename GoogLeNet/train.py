import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning.callbacks import (
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy

import datamodule
from model import GoogLeNet

sys.path.insert(1, os.path.abspath('..'))
import utils


class GoogLeNetModel(pl.LightningModule):

    def __init__(
            self,
            image_channels: int,
            n_classes: int,
            lr: float,
            epochs: int,
            lr_step: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = GoogLeNet(
            self.hparams.image_channels,
            self.hparams.n_classes,
        )
        self.model.initialize_weights()

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.epochs = epochs
        self.lr_step = lr_step

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
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
        )
        scheduler_dict = {
            'scheduler': MultiStepLR(
                optimizer,
                milestones=utils.get_milestones(
                    epoch=self.epochs,
                    step=self.lr_step
                ),
                gamma=0.92,
                verbose=True,
            )
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


def train():
    # Hyperparameters
    config = utils.get_config()
    hparams = config.hparams
    seed_everything(config.seed)

    # Dataloader
    dm = datamodule.CIFAR10DataModule(
        config.data_dir,
        image_size=config.image_size,
        batch_size=hparams.batch_size,
    )

    # Model
    googlenet = GoogLeNetModel(
        image_channels=config.image_channels,
        n_classes=config.n_classes,
        lr=hparams.lr,
        epochs=hparams.epochs,
        lr_step=hparams.lr_step,
    )

    # Logger
    wandb_logger = WandbLogger(
        name=f'{config.project_name}-{config.dataset}',
        project=config.project_name,
        save_dir=config.save_dir,
        log_model='all',
    )
    wandb_logger.experiment.config.update(hparams)
    wandb_logger.watch(googlenet, log='all', log_freq=100)

    # Trainer setting
    callbacks = [
        EarlyStopping(
            monitor='val_acc',
            min_delta=0.00,
            patience=3,
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
    trainer.fit(googlenet, datamodule=dm)
    trainer.test(googlenet, datamodule=dm)

    # Finish
    wandb_logger.experiment.unwatch(googlenet)

    # Model to Torchscript
    script = googlenet.to_torchscript()
    torch.jit.save(script, config.torchscript_model_save_path)
    wandb_logger.experiment.save(config.torchscript_model_save_path)


if __name__=='__main__':
    train()
