import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import yaml
from mlflow.models import infer_signature

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        predicted = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        predicted = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    config_path = Path("/app/configs/training_config.yaml")

    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    config = load_config(str(config_path))
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("pytorch-cifar10")

    mlflow.start_run()
    mlflow.log_params({
        "architecture": config["model"]["architecture"],
        "num_classes": config["model"]["num_classes"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "early_stopping_patience": config["training"]["early_stopping_patience"],
        "dataset": config["data"]["dataset"],
        "train_samples": config["data"]["train_samples"],
        "val_samples": config["data"]["val_samples"],
    })

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(json.dumps({
        "event": "training_started",
        "device": str(device),
        "architecture": config["model"]["architecture"],
    }), flush=True)

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        train_samples=config["data"]["train_samples"],
        val_samples=config["data"]["val_samples"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / config["output"]["model_name"]

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }

        print(json.dumps(log_entry), flush=True)
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            step=epoch + 1,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )
            mlflow.log_artifact(
                str(checkpoint_path),
                artifact_path="checkpoint",
            )
            print(json.dumps({
                "event": "checkpoint_saved",
                "path": str(checkpoint_path),
            }), flush=True)

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(json.dumps({
                    "event": "early_stopping",
                    "epoch": epoch + 1,
                }), flush=True)
                break
    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    input_example = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        output_example = model(input_example)

    signature = infer_signature(
        input_example.numpy(),
        output_example.numpy(),
    )

    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name="CIFAR10_SmallCNN",
        signature=signature,
        input_example=input_example.numpy(),
    )

    mlflow.log_metric("best_val_loss", best_val_loss)

    mlflow.end_run()

    print(json.dumps({
        "event": "training_complete",
        "best_val_loss": round(best_val_loss, 4),
    }), flush=True)


if __name__ == "__main__":
    main()
