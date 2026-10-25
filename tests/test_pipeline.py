import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from ml_pipeline.data import load_iris_data, validate_data
from ml_pipeline.orchestrator import run_pipeline


class DataValidationTests(unittest.TestCase):
    def test_iris_data_passes_validation(self) -> None:
        features, target, _ = load_iris_data()

        summary = validate_data(features, target)

        self.assertEqual(summary, {"rows": 150, "features": 4, "classes": 3})

    def test_validation_rejects_missing_values(self) -> None:
        features, target, _ = load_iris_data()
        features = features.copy()
        features.iloc[0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "missing values"):
            validate_data(features, target)

    def test_validation_rejects_mismatched_row_counts(self) -> None:
        features, target, _ = load_iris_data()

        with self.assertRaisesRegex(ValueError, "row counts do not match"):
            validate_data(features, target.iloc[:-1])


class PipelineTests(unittest.TestCase):
    def test_pipeline_trains_evaluates_and_saves_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "iris_model.joblib"
            with patch("ml_pipeline.orchestrator.log_experiment") as log_experiment:
                result = run_pipeline(model_path=model_path)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["data"]["rows"], 150)
            self.assertEqual(result["split"], {"train_rows": 120, "test_rows": 30})
            self.assertGreaterEqual(result["metrics"]["accuracy"], 0.9)
            self.assertTrue(model_path.is_file())
            self.assertIn("model", joblib.load(model_path))
            log_experiment.assert_called_once()


if __name__ == "__main__":
    unittest.main()