from typing import *

import torch.nn as nn
import torch.optim as optim

from models.LitBase import LitBase

from .models import GoogLeNet


class LitGoogLeNet(LitBase):
    def __init__(self, args: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = GoogLeNet(
            image_channels=self.hparams.image_channels,
            num_classes=self.hparams.num_classes,
            dropout_rate=self.hparams.dropout_rate,
        )
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler_dict = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.hparams.scheduler_mode,
                factor=self.hparams.scheduler_factor,
                patience=self.hparams.scheduler_patience,
                verbose=True,
            ),
            "monitor": self.hparams.scheduler_monitor,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
