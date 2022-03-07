from typing import *

import torch.nn as nn
import torch.optim as optim
from torchmetrics import functional as tmf

from models.LitBase import LitBase

from .models import *


class LitInceptionV3(LitBase):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = Inception_v3(
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

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        logit, aux_logits = self(x)

        aux_loss = self.loss(aux_logits, y)
        main_loss = self.loss(logit, y)

        loss = main_loss * self.hparams.loss_w + aux_loss * self.hparams.aux_loss_w
        self.log_dict(
            {
                "train/loss": loss,
                "train/acc": tmf.accuracy(logit, y),
                "train/acc_top_3": tmf.accuracy(logit, y, top_k=3),
                "train/acc_top_5": tmf.accuracy(logit, y, top_k=5),
            },
            prog_bar=True,
        )

        return loss
