from typing import *

import torch.nn as nn
import torch.optim as optim

from models.LitBase import LitBase
from .models import LeNet5


class LitLeNet5(LitBase):
    def __init__(self, args: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = LeNet5(
            image_channels=self.hparams.image_channels,
            num_classes=self.hparams.num_classes,
        )
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
