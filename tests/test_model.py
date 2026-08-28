import torch

from src.model import SmallCNN, get_model


def test_small_cnn_output_shape():
    model = SmallCNN(num_classes=10)

    inputs = torch.randn(4, 3, 32, 32)
    outputs = model(inputs)

    assert outputs.shape == (4, 10)


def test_get_model_returns_small_cnn():
    model = get_model(
        architecture="small_cnn",
        num_classes=10,
    )

    assert isinstance(model, SmallCNN)
