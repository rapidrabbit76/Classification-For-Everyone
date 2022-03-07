from typing import *
from abc import ABCMeta, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import functional as tmf

_batch_type = Tuple[Tensor, Tensor]


class LitBase(pl.LightningModule, metaclass=ABCMeta):
    @abstractmethod
    def configure_optimizers(self):
        return super().configure_optimizers()

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

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _common_step(self, batch: _batch_type) -> _batch_type:
        x, y = batch
        logit = self(x)
        return (logit, y)

    def training_step(self, batch: _batch_type, batch_idx: int) -> Tensor:
        logit, y = self._common_step(batch)
        loss = self.loss(logit, y)
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

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch)

    def test_step(self, batch, batch_idx):
        return self._common_step(batch)

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

    def test_epoch_end(self, outputs: List[Tensor]) -> Dict[str, Union[Tensor, float]]:
        return self._validation_test_common_epoch_end(outputs, "test")
