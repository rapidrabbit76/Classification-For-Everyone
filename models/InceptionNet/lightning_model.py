from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import functional as tmf

from .models import *

_batch_type = Tuple[Tensor, Tensor]
_step_return_type = Tuple[Dict[str, Tensor], Tensor]


class LitInceptionV3(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.model = Inception_v3(
            image_channels=self.hparams.image_channels,
            num_classes=self.hparams.num_classes,
        )

        # Metrics
        self.loss = nn.CrossEntropyLoss()
        self.validation_step = self._validation_test_common_step
        self.test_step = self._validation_test_common_step

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

    def forward(self, x: Tensor) -> MODEL_RETURN_TYPE:
        return self.model(x)

    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def training_step(
        self,
        batch: _batch_type,
        batch_idx: int,
    ) -> Tensor:
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

    def _validation_test_common_step(
        self,
        batch: _batch_type,
        batch_idx: int,
    ) -> _batch_type:
        x, y = batch
        logit = self(x)
        return (logit, y)

    def _validation_test_common_epoch_end(
        self,
        outputs: List[Tuple[Tensor, Tensor]],
        mode: str,
    ) -> Dict[str, Union[Tensor, float]]:
        # concatnation every loggit, target
        logits = torch.cat([logit for logit, _ in outputs])
        targets = torch.cat([target for _, target in outputs])

        # calcuration metrics
        metric_dict = {
            f"{mode}/loss": self.loss(logits, targets),
            f"{mode}/acc": tmf.accuracy(logits, targets),
            f"{mode}/acc_top_3": tmf.accuracy(logits, targets, top_k=3),
            f"{mode}/acc_top_5": tmf.accuracy(logits, targets, top_k=5),
        }

        self.log_dict(metric_dict, prog_bar=True)
        return metric_dict

    def validation_epoch_end(
        self, outputs: List[Tensor]
    ) -> Dict[str, Union[Tensor, float]]:

        return self._validation_test_common_epoch_end(outputs, "val")

    def test_epoch_end(
        self, outputs: List[Tensor]
    ) -> Dict[str, Union[torch.Tensor, float]]:

        return self._validation_test_common_epoch_end(outputs, "test")
