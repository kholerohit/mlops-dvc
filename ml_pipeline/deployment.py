from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, metadata: dict[str, Any], model_path: Path) -> Path:
    """Package the fitted model and its inference metadata for local serving."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, **metadata}, model_path)
    return model_path


def load_model(model_path: Path) -> dict[str, Any]:
    """Load a model artifact produced by this demo pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)
