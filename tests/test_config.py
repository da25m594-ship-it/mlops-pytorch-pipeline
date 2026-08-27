from pathlib import Path
import yaml


def test_training_config_exists():
    config_path = Path("configs/train.yaml")
    assert config_path.exists()


def test_training_config_loads():
    with open("configs/train.yaml") as f:
        config = yaml.safe_load(f)

    assert config["model"]["num_classes"] == 10
    assert config["training"]["epochs"] == 2
