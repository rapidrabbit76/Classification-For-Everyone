import pytest
from models.EfficientNetV1 import *

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
    params=[
        EfficientNet_b0,
        EfficientNet_b1,
        EfficientNet_b2,
        EfficientNet_b3,
        EfficientNet_b4,
        EfficientNet_b5,
        EfficientNet_b6,
        EfficientNet_b7,
    ],
)
def model(request, config):
    return request.param(
        config.image_channels,
        config.num_classes,
    )
