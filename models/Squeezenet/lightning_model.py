from typing import *

import torch.nn as nn
import torch.optim as optim

from models.LitBase import LitBase

from models.Squeezenet.models import SqueezeNet


class LitSqueezeNet(LitBase):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = SqueezeNet(
            self.hparams.image_channels,
            self.hparams.num_classes,
        )
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
