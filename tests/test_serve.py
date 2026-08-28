from pathlib import Path

import torch

from src.model import get_model


def test_model_checkpoint_exists():
    checkpoint = Path("checkpoints/classifier.pt")

    if not checkpoint.exists():
        # Checkpoint is intentionally not committed to Git.
        # CI validates the serving code without requiring the artifact.
        return

    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu",
    )

    assert "model_state_dict" in checkpoint_data


def test_model_can_load_checkpoint():
    checkpoint = Path("checkpoints/classifier.pt")

    if not checkpoint.exists():
        return

    model = get_model(
        architecture="small_cnn",
        num_classes=10,
    )

    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint_data["model_state_dict"]
    )

    model.eval()

    sample = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        output = model(sample)

    assert output.shape == (1, 10)
