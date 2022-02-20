import pytest


def test_model_inference(config, datamodule):
    train_loader = datamodule.train_dataloader()

    for batch in train_loader:
        image, target = batch
