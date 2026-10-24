from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from ml_pipeline.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from ml_pipeline.data import load_iris_data, validate_data
from ml_pipeline.deployment import save_model
from ml_pipeline.evaluation import evaluate_model
from ml_pipeline.training import train_model


def run_pipeline(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Run data ingestion through evaluation and save a deployable artifact."""
    features, target, metadata = load_iris_data()
    data_summary = validate_data(features, target)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    model = train_model(x_train, y_train, random_state=RANDOM_STATE)
    metrics = evaluate_model(model, x_test, y_test)
    metadata["metrics"] = metrics
    artifact_path = save_model(model, metadata, model_path)
    return {
        "status": "completed",
        "data": data_summary,
        "split": {"train_rows": int(len(x_train)), "test_rows": int(len(x_test))},
        "metrics": metrics,
        "model_path": str(artifact_path),
    }
