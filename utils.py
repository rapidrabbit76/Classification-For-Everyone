import os
import pytorch_lightning as pl
import yaml
import numpy as np
from easydict import EasyDict
import torch


def get_config(config_path: str = "./config.yaml"):
    return EasyDict(
        yaml.load(
            open(config_path, "r"),
            Loader=yaml.FullLoader,
        )
    )


def get_milestones(epoch: int, step: int):
    step_size = np.trunc(epoch / step)
    milestone = list(range(0, int(step_size+1), step))
    return milestone[1:]


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