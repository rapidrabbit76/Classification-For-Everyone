from typing import Any, Dict, List, Optional, Type, Union, Tuple, Final

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchmetrics import functional as tmf

from .models import MNASNet


__all__ = ["LitMNASNet"]

_batch_type: Final[Type] = Tuple[torch.Tensor, torch.Tensor]


class LitMNASNet(pl.LightningModule):
    def __init__(
            self,
            config: Dict[str, Union[str, int, float, bool]],
    ):
        super().__init__()
        self.save_hyperparameters(config)

        self.model = MNASNet(
            image_channels=self.hparams.image_channels,
            num_classes=self.hparams.num_classes,
            dropout_rate=self.hparams.dropout_rate
        )

        # Metrics
        self.loss = nn.CrossEntropyLoss()

        self.validation_step = self._validation_test_common_step
        self.test_step = self._validation_test_common_step

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler_dict = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode=self.hparams.scheduler_mode,
                factor=self.hparams.scheduler_factor,
                patience=self.hparams.scheduler_patience,
                verbose=True,
            ),
            'monitor': self.hparams.scheduler_monitor,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
            self,
            batch: _batch_type,
            batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        logit = self(x)
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
            outputs: List[Tuple[torch.Tensor, torch.Tensor]],
            mode: str,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        # concatnation every logit, target
        logits = torch.cat([logit for logit, _ in outputs])
        targets = torch.cat([target for _, target in outputs])

        # calculation metrics
        metric_dict= {
            f"{mode}/loss": self.loss(logits, targets),
            f"{mode}/acc": tmf.accuracy(logits, targets),
            f"{mode}/acc_top_3": tmf.accuracy(logits, targets, top_k=3),
            f"{mode}/acc_top_5": tmf.accuracy(logits, targets, top_k=5),
        }

        self.log_dict(metric_dict, prog_bar=True)
        return metric_dict

    def validation_epoch_end(
            self,
            outputs: List[torch.Tensor],
    ) -> Dict[str, Union[torch.Tensor, float]]:
        return self._validation_test_common_epoch_end(outputs, "val")

    def test_epoch_end(
            self,
            outputs: List[torch.Tensor],
    ) -> Dict[str, Union[torch.Tensor, float]]:
        return self._validation_test_common_epoch_end(outputs, "test")