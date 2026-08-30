# MLOps PyTorch Pipeline

**Student:** Nambari Lokanadham  
**Roll No:** DA25M594

GitHub Repository:  
https://github.com/da25m594-ship-it/mlops-pytorch-pipeline

---

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for a PyTorch CIFAR-10 image classification model.

The pipeline covers:

- PyTorch model training
- Experiment tracking using MLflow
- MLflow Model Registry
- Model deployment utility
- Docker-based model serving
- FastAPI prediction API
- Kubernetes deployment using Minikube
- Automated testing using pytest
- Continuous Integration using GitHub Actions
- Git-based development using feature branches and pull requests

---

## 2. Architecture

```mermaid
flowchart LR

    A[CIFAR-10 Dataset] --> B[PyTorch Training]

    B --> C[MLflow Experiment Tracking]

    B --> D[Checkpoint<br/>classifier.pt]

    C --> E[MLflow Model Registry]

    E --> F[Registered Model<br/>CIFAR10_SmallCNN v1]

    F --> G[deploy_model.py]

    G --> D

    D --> H[Docker Image<br/>mlops-pytorch-serve]

    H --> I[FastAPI Serving]

    I --> J[/health]
    I --> K[/predict]

    H --> L[Kubernetes / Minikube]

    L --> M[Deployment]
    L --> N[NodePort Service]

    O[GitHub Repository] --> P[Pull Requests]
    P --> Q[GitHub Actions CI]
    Q --> R[pytest]

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.10 |
| Deep Learning | PyTorch 2.5.1 |
| Dataset | CIFAR-10 |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| API | FastAPI |
| Containerization | Docker |
| Orchestration | Kubernetes |
| Local Kubernetes | Minikube |
| Testing | pytest |
| CI | GitHub Actions |
| Version Control | Git / GitHub |

## 4. Repository Structure

```text
mlops-pytorch-pipeline/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   └── training_config.yaml
│
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── requirements/
│   ├── train.txt
│   └── serve.txt
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── serve.py
│   └── deploy_model.py
│
├── tests/
│   ├── test_config.py
│   ├── test_model.py
│   ├── test_serve.py
│   └── test_deploy_model.py
│
├── data/
├── checkpoints/
├── README.md
└── .gitignore


---

## 5. Model

```markdown
## 5. Model

The project uses a small convolutional neural network (`SmallCNN`) for CIFAR-10 image classification.

Input:

- RGB image
- Size: 3 × 32 × 32

Output:

- 10 class scores corresponding to the CIFAR-10 classes.

The serving API applies CIFAR-10 normalization before inference.

## 6. Training

The training pipeline is implemented in:

`src/train.py`

The training process:

1. Loads the CIFAR-10 dataset.
2. Creates training and validation data loaders.
3. Builds the SmallCNN model.
4. Trains the model on CPU.
5. Evaluates validation loss and accuracy.
6. Saves the best model checkpoint.
7. Logs parameters and metrics to MLflow.
8. Logs the PyTorch model to MLflow.
9. Registers the model as `CIFAR10_SmallCNN`.

## 7. MLflow Experiment Tracking

MLflow is used to track:

- Training parameters
- Training loss
- Training accuracy
- Validation loss
- Validation accuracy
- Best validation loss
- Model artifacts

The local MLflow tracking directory is:

`mlruns/`

The experiment used in the project is:

`pytorch-cifar10`

## 8. MLflow Model Registry

The trained PyTorch model is registered in MLflow Model Registry.

Registered model:

`CIFAR10_SmallCNN`

Validated model version:

- Version: 1
- Status: READY

The registered model can be loaded using:

```python
mlflow.pytorch.load_model(
    "models:/CIFAR10_SmallCNN/1"
)


---

## 9. Model Deployment Utility

```markdown
## 9. Model Deployment Utility

The project provides:

`src/deploy_model.py`

This utility:

1. Loads a registered model from MLflow.
2. Selects the requested model version.
3. Converts the registered model into the application's checkpoint format.
4. Saves the checkpoint as `classifier.pt`.

Example:

```bash
python src/deploy_model.py \
    --model-name CIFAR10_SmallCNN \
    --version 1 \
    --output checkpoints/classifier.pt


---

## 10. Docker

```markdown
## 10. Docker

Two Docker images are used.

### Training Image

`mlops-pytorch-train:dev`

Build:

```bash
docker build \
    -f docker/Dockerfile.train \
    -t mlops-pytorch-train:dev .


---

## 11. FastAPI Model Serving

```markdown
## 11. FastAPI Model Serving

The trained model is served using FastAPI.

Run the serving container:

```bash
docker run -d \
    --name mlops-pytorch-serve \
    -p 8000:8000 \
    -v "$PWD/checkpoints:/app/checkpoints:ro" \
    mlops-pytorch-serve:dev


---

## 12. Health Check

```markdown
## 12. Health Check

The API provides a health endpoint:

`GET /health`

Example:

```bash
curl http://localhost:8000/health


---

## 13. Prediction API

```markdown
## 13. Prediction API

The prediction endpoint is:

`POST /predict`

Example:

```bash
curl -X POST \
    -F "image=@data/test_image.png" \
    http://localhost:8000/predict

{
  "predicted_class": 6,
  "class_probabilities": [...]
}


---

## 14. Kubernetes Deployment

```markdown
## 14. Kubernetes Deployment

The serving application can be deployed using Kubernetes and Minikube.

Start Minikube:

```bash
minikube start --driver=docker



---

## 15. Kubernetes Health and Readiness

```markdown
## 15. Kubernetes Health and Readiness

The Kubernetes Deployment uses the FastAPI `/health` endpoint for:

- Readiness probe
- Liveness probe

The model checkpoint is mounted into the Pod at:

`/app/checkpoints/classifier.pt`

The serving application loads the model during FastAPI startup.

The validated Kubernetes Pod reached:

`READY 1/1`

with:

`STATUS Running`

## 16. Testing

The project uses pytest for automated testing.

Run the tests:

```bash
PYTHONPATH=. pytest -q


---

## 17. Continuous Integration

```markdown
## 17. Continuous Integration

GitHub Actions is configured in:

`.github/workflows/ci.yml`

The CI pipeline:

1. Checks out the repository.
2. Sets up Python 3.10.
3. Installs test dependencies.
4. Installs MLflow.
5. Runs pytest.
6. Reports the test result.

The final GitHub Actions CI checks completed successfully with:

`7 passed`

## 18. Git Branching and Pull Requests

The project was developed using feature branches and pull requests.

At least four meaningful pull requests were successfully merged into the `develop` branch.

### PR #3 — Model Serving to Kubernetes

Implemented Kubernetes deployment and service configuration for the PyTorch model serving application.

### PR #4 — MLflow Experiment Tracking

Implemented MLflow experiment tracking for training parameters and model performance metrics.

### PR #5 — MLflow Model Registry

Implemented PyTorch model logging and registration using MLflow Model Registry.

Registered model:

`CIFAR10_SmallCNN`

### PR #6 — MLflow Model Deployment Utility

Implemented the MLflow model deployment utility:

`src/deploy_model.py`

The utility retrieves a registered model version and converts it into the checkpoint format required by the serving application.

Unit tests were also added for the deployment utility.

All four PRs were merged successfully into `develop`.

## 19. Complete MLOps Workflow

The implemented workflow is:

```text
PyTorch Training
       ↓
MLflow Experiment Tracking
       ↓
MLflow Model Registry
       ↓
Registered Model
       ↓
deploy_model.py
       ↓
classifier.pt
       ↓
Docker
       ↓
FastAPI
       ↓
Kubernetes / Minikube
       ↓
Prediction API


---

## 20. Validation Summary

```markdown
## 20. Validation Summary

The implementation was validated at multiple levels.

### Code Validation

```text
python -m py_compile
git diff --check


---

## 21. Conclusion

```markdown
## 21. Conclusion

This project demonstrates an end-to-end MLOps workflow for a PyTorch CIFAR-10 image classification model.

The implementation integrates:

- PyTorch model training
- MLflow experiment tracking
- MLflow Model Registry
- Model deployment automation
- Docker containerization
- FastAPI model serving
- Kubernetes deployment
- Automated testing
- GitHub Actions CI
- Git feature branches and pull requests

The project demonstrates the complete lifecycle from model development and experiment tracking through model registration, deployment, API serving, and Kubernetes orchestration.

## GitHub Repository

[https://github.com/da25m594-ship-it/mlops-pytorch-pipeline](https://github.com/da25m594-ship-it/mlops-pytorch-pipeline)


