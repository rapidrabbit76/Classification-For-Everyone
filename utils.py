from easydict import EasyDict
import yaml
from pytorch_lightning.loggers import WandbLogger


def get_config(config_path: str = "config.yml"):
    return EasyDict(
        yaml.load(
            open(config_path, "r"),
            Loader=yaml.FullLoader,
        )
    )
