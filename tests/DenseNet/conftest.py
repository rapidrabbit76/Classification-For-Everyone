import pytest
from models.DenseNet.models import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    DenseNet265,
)

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
            "image_size": 224,
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
    params=[DenseNet121, DenseNet201, DenseNet169, DenseNet265],
)
def model(request, config):
    return request.param(
        config.image_channels,
        config.num_classes,
    )
