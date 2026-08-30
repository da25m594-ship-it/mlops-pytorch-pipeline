import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch


def deploy_model(
    model_name: str,
    version: str,
    output_path: str,
) -> None:
    model_uri = f"models:/{model_name}/{version}"

    print(f"Loading registered model: {model_uri}", flush=True)

    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        output,
    )

    print(
        f"Model deployed to: {output}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Deploy an MLflow registered PyTorch model."
    )

    parser.add_argument(
        "--model-name",
        required=True,
    )

    parser.add_argument(
        "--version",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="checkpoints/classifier.pt",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(
        "file:/app/mlruns"
    )

    deploy_model(
        model_name=args.model_name,
        version=args.version,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
