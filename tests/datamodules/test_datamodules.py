import pytest


def test_model_inference(config, datamodule):
    train_loader = datamodule.train_dataloader()
    w = h = config.image_size
    image_shape = [1, config.image_channels, w, h]

    for batch in train_loader:
        image, target = batch
        break
