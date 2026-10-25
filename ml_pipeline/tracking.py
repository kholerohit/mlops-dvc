from typing import Any

import dagshub
import mlflow
import mlflow.sklearn


DAGSHUB_TRACKING_URI = "https://dagshub.com/kholerohit/mlops-dvc.mlflow"
EXPERIMENT_NAME = "iris-classification"


def log_experiment(
    model: Any,
    parameters: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Log one completed training run and its sklearn model to DagsHub."""
    dagshub.init(repo_owner="kholerohit", repo_name="mlops-dvc", mlflow=True)
    mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")