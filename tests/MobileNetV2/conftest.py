import pytest
from models.MobileNetV2.models import *
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
    params=[
        MobileNetV2_05,
        MobileNetV2_075,
        MobileNetV2_10,
    ],
)
def model(request, config):
    return request.param(
        config.image_channels,
        config.num_classes,
    )
