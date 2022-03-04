import pytest


@pytest.mark.parametrize("mode", ["train", "test"])
def test_model_inference(config, model, mode, batch):
    model = model if mode == "train" else model.eval()
    image, target = batch
    x = model(image)
    size = x.shape

    assert size[1] == config.num_classes
