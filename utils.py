from easydict import EasyDict
import yaml
import numpy as np


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