from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_transformer() -> StandardScaler:
    """Create the feature transformation fitted only on the training split."""
    return StandardScaler()


def build_model_pipeline(classifier: object) -> Pipeline:
    """Combine preprocessing and a classifier to prevent train/test leakage."""
    return Pipeline(
        steps=[
            ("scaler", build_transformer()),
            ("classifier", classifier),
        ]
    )
