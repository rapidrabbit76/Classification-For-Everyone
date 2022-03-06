from typing import *

import torch.nn as nn
import torch.optim as optim

from models.LitBase import LitBase


from .models import ShuffleNetV2


class LitShuffleNetV2(LitBase):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = ShuffleNetV2(
            model_type=self.hparams.model_type,
            image_channels=self.hparams.image_channels,
            num_classes=self.hparams.num_classes,
        )
        self.loss = nn.CrossEntropyLoss()

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
            "scheduler": scheduler,
            "interval": self.hparams.scheduler_interval,
            "frequency": self.hparams.scheduler_frequency,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }
