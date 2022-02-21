import pytest
from easydict import EasyDict
from typing import Dict, Any

from datamodules import *
from transforms import *


@pytest.fixture(scope="module")
def config() -> Dict[str, Any]:
    return EasyDict(
        {
            "root_dir": "DATASET",
            "batch_size": 1,
            "image_channels": 3,
            "image_size": 224,
        }
    )


@pytest.fixture(scope="module")
def train_transforms(config):
    image_shape = [
        config.image_channels,
        config.image_size,
        config.image_size,
    ]
    return BaseTransforms(
        image_shape=image_shape,
        train=False,
    )


@pytest.fixture(
    scope="module",
    params=[
        # MnistDataModule,
        CIFAR10DataModule,
        # CIFAR100DataModule,
    ],
)
def datamodule(request, config, train_transforms):
    dm = request.param(
        root_dir=config.root_dir,
        train_transforms=train_transforms,
        test_transforms=train_transforms,
        batch_size=config.batch_size,
    )
    dm.setup("fit")
    return dm
