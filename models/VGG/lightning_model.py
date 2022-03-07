from typing import *

import torch.nn as nn
import torch.optim as optim

from models.LitBase import LitBase

from .models import VGGModel


class LitVGG(LitBase):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = VGGModel(
            model_type=self.hparams.model_type,
            image_channels=self.hparams.image_channels,
            num_classes=self.hparams.num_classes,
            dropout_rate=self.hparams.dropout_rate,
        )
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
