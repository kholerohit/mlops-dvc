from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "iris_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
