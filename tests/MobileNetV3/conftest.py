import pytest
from models.MobileNetV3.models import *
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
        torch.rand(2, c, w, h),
        torch.rand(2),
    )


@pytest.fixture(
    scope="module",
    params=[MobileNetV3_s, MobileNetV3_l],
)
def model(request, config):
    return request.param(
        config.image_channels,
        config.num_classes,
    )
