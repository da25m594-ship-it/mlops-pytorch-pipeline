
import torch

from src.model import get_model
from src.deploy_model import deploy_model


def test_deploy_model_creates_checkpoint(tmp_path, monkeypatch):
    class FakeModel:
        def __init__(self):
            self._model = get_model(
                architecture="small_cnn",
                num_classes=10,
            )

        def eval(self):
            self._model.eval()
            return self

        def state_dict(self):
            return self._model.state_dict()

    fake_model = FakeModel()

    monkeypatch.setattr(
        "src.deploy_model.mlflow.pytorch.load_model",
        lambda uri: fake_model,
    )

    output_path = tmp_path / "classifier.pt"

    deploy_model(
        model_name="CIFAR10_SmallCNN",
        version="1",
        output_path=str(output_path),
    )

    assert output_path.exists()

    checkpoint = torch.load(
        output_path,
        map_location="cpu",
        weights_only=True,
    )

    assert "model_state_dict" in checkpoint
    assert len(checkpoint["model_state_dict"]) == 8
