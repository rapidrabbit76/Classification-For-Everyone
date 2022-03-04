import pytest


def test_model_inference(config, datamodule):
    train_loader = datamodule.train_dataloader()
    w = h = config.image_size
    image_shape = [config.batch_size, config.image_channels, w, h]
    target_shape = [config.batch_size]
    image, target = next(iter(train_loader))

    assert list(image.size()) == image_shape
    assert list(target.size()) == target_shape
