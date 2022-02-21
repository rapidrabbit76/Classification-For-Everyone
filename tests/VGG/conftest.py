import pytest
from models.VGG.models import VGG11, VGG13, VGG16, VGG19
from easydict import EasyDict
import torch


@pytest.fixture(
    scope="module",
)
def config():
    return EasyDict(
        {
            "image_channals": 3,
            "num_classes": 10,
            "dropout_rate": 0.5,
            "image_size": 224,
        }
    )


@pytest.fixture(
    scope="module",
)
def batch(config):
    c = config.image_channals
    w = h = config.image_size
    return (
        torch.rand(1, c, w, h),
        torch.rand(1),
    )


@pytest.fixture(
    scope="module",
    params=[VGG11, VGG13, VGG16, VGG19],
)
def model(request, config):
    return request.param(
        config.image_channals,
        config.num_classes,
        config.dropout_rate,
    )
