from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_pipeline.transformation import build_model_pipeline


def train_model(
    features: Any,
    target: Any,
    random_state: int,
    max_iter: int = 1000,
) -> Pipeline:
    """Fit a scaled logistic-regression classifier on the training data."""
    model = build_model_pipeline(
        LogisticRegression(max_iter=max_iter, random_state=random_state)
    )
    model.fit(features, target)
    return model
