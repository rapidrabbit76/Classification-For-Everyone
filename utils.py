import os

import pytorch_lightning as pl
import yaml
from easydict import EasyDict
import torch


def get_config(config_path: str = "config.yaml"):
    return EasyDict(
        yaml.load(
            open(config_path, "r"),
            Loader=yaml.FullLoader,
        )
    )


def model_save(
    model: pl.LightningModule,
    save_dir: str,
    model_name: str,
) -> str:
    script = model.to_torchscript()
    saved_model_path = os.path.join(
        save_dir,
        f"{model_name}.pt",
    )
    torch.jit.save(script, saved_model_path)
    return saved_model_path
