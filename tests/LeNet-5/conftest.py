import pytest
from models.LeNet5.models import *
from easydict import EasyDict
import torch


@pytest.fixture(
    scope="module",
)
def config():
    return EasyDict(
        {
            "image_channels": 3,
            "num_classes": 10,
            "dropout_rate": 0.5,
            "image_size": 32,
        }
    )


@pytest.fixture(
    scope="module",
)
def batch(config):
    c = config.image_channels
    w = h = config.image_size
    return (
        torch.rand(1, c, w, h),
        torch.rand(1),
    )


@pytest.fixture(
    scope="module",
)
def model(config):
    return LeNet5(
        config.image_channels,
        config.num_classes,
    )
