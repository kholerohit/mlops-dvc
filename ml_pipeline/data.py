from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def load_iris_data() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load the bundled Iris dataset without a network or external data file."""
    dataset = load_iris(as_frame=True)
    features = dataset.data
    target = dataset.target
    metadata = {
        "feature_names": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
        "description": "Iris flower measurements bundled with scikit-learn",
    }
    return features, target, metadata


def validate_data(features: pd.DataFrame, target: pd.Series) -> dict[str, int]:
    """Check the input table before it is split or used for training."""
    if features.empty:
        raise ValueError("The feature table is empty.")
    if len(features) != len(target):
        raise ValueError("Feature and target row counts do not match.")
    if features.columns.empty or not features.columns.is_unique:
        raise ValueError("Feature columns must be present and unique.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("All features must be numeric.")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("The dataset contains missing values.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Features contain non-finite values.")
    if target.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")
    return {"rows": int(len(features)), "features": int(features.shape[1]), "classes": int(target.nunique())}
