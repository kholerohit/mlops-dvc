import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_pipeline.config import PROJECT_ROOT
from ml_pipeline.data import load_iris_data, validate_data
from ml_pipeline.deployment import save_model
from ml_pipeline.evaluation import evaluate_model
from ml_pipeline.training import train_model


RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "iris.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "iris.csv"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "reports" / "validation.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
TRAINED_MODEL_PATH = PROJECT_ROOT / "artifacts" / "trained_model.joblib"
DEPLOYED_MODEL_PATH = PROJECT_ROOT / "artifacts" / "iris_model.joblib"
TARGET_COLUMN = "target"


def _read_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(path)
    if TARGET_COLUMN not in dataset:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")
    return dataset.drop(columns=[TARGET_COLUMN]), dataset[TARGET_COLUMN]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def ingest() -> None:
    features, target, _ = load_iris_data()
    dataset = features.copy()
    dataset[TARGET_COLUMN] = target
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(RAW_DATA_PATH, index=False)
    print(f"Ingested {len(dataset)} Iris rows into {RAW_DATA_PATH.relative_to(PROJECT_ROOT)}")


def validate() -> None:
    features, target = _read_dataset(RAW_DATA_PATH)
    summary = validate_data(features, target)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([features, target.rename(TARGET_COLUMN)], axis=1).to_csv(
        PROCESSED_DATA_PATH, index=False
    )
    _write_json(VALIDATION_REPORT_PATH, {"status": "passed", **summary})
    print(f"Validation passed: {summary}")


def split(test_size: float, random_state: int) -> None:
    features, target = _read_dataset(PROCESSED_DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([x_train, y_train.rename(TARGET_COLUMN)], axis=1).to_csv(
        TRAIN_DATA_PATH, index=False
    )
    pd.concat([x_test, y_test.rename(TARGET_COLUMN)], axis=1).to_csv(
        TEST_DATA_PATH, index=False
    )
    print(f"Created train/test split: {len(x_train)} / {len(x_test)} rows")


def train(random_state: int, max_iter: int) -> None:
    features, target = _read_dataset(TRAIN_DATA_PATH)
    model = train_model(features, target, random_state=random_state, max_iter=max_iter)
    TRAINED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, TRAINED_MODEL_PATH)
    print(f"Saved trained model to {TRAINED_MODEL_PATH.relative_to(PROJECT_ROOT)}")


def evaluate() -> None:
    features, target = _read_dataset(TEST_DATA_PATH)
    model = joblib.load(TRAINED_MODEL_PATH)
    metrics = evaluate_model(model, features, target)
    _write_json(METRICS_PATH, metrics)
    print(f"Held-out metrics: {metrics}")


def deploy() -> None:
    model = joblib.load(TRAINED_MODEL_PATH)
    _, _, metadata = load_iris_data()
    metadata["metrics"] = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    artifact_path = save_model(model, metadata, DEPLOYED_MODEL_PATH)
    print(f"Packaged model for API serving at {artifact_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an individual DVC pipeline stage.")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("ingest")
    subparsers.add_parser("validate")

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--test-size", type=float, required=True)
    split_parser.add_argument("--random-state", type=int, required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--random-state", type=int, required=True)
    train_parser.add_argument("--max-iter", type=int, required=True)
    subparsers.add_parser("evaluate")
    subparsers.add_parser("deploy")

    args = parser.parse_args()
    if args.stage == "ingest":
        ingest()
    elif args.stage == "validate":
        validate()
    elif args.stage == "split":
        split(args.test_size, args.random_state)
    elif args.stage == "train":
        train(args.random_state, args.max_iter)
    elif args.stage == "evaluate":
        evaluate()
    elif args.stage == "deploy":
        deploy()


if __name__ == "__main__":
    main()