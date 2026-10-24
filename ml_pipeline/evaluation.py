from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(model: Any, features: Any, target: Any) -> dict[str, float]:
    """Calculate standard classification metrics on the held-out split."""
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision_macro": float(precision_score(target, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(target, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(target, predictions, average="macro", zero_division=0)),
    }
