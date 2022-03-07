import pytest
from models.ShuffleNet.models import *

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
    params=[ShuffleNetV2_x05, ShuffleNetV2_x10, ShuffleNetV2_x15, ShuffleNetV2_x20],
)
def model(request, config):
    return request.param(
        config.image_channels,
        config.num_classes,
    )
