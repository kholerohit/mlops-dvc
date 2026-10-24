# Modular ML Pipeline Demo

A small end-to-end classification project. It uses the Iris dataset bundled with
scikit-learn, so no external data file or network download is needed.

## Pipeline stages

- **Data ingestion:** load Iris from `sklearn.datasets`.
- **Validation:** check row counts, numeric and finite features, missing values,
	and target classes.
- **Transformation:** fit `StandardScaler` inside a scikit-learn pipeline using
	only the training split.
- **Training:** fit logistic regression with a reproducible stratified split.
- **Evaluation:** report accuracy, macro precision, macro recall, and macro F1 on
	the held-out test split.
- **Deployment:** save the fitted pipeline and inference metadata to
	`artifacts/iris_model.joblib`.
- **Frontend/API:** serve a small prediction interface and JSON endpoints with
	FastAPI.

## Setup

```bash
conda activate mlops
python -m pip install -r requirements.txt
```

## Run the complete pipeline

```bash
conda activate mlops
python demo.py
```

The command prints the validation summary, split sizes, evaluation metrics, and
artifact location.

## Run with DVC

The `dvc.yaml` pipeline separates ingestion, validation, data splitting, model
training, evaluation, and deployment packaging. Run all stages from the project
root while `mlops` is active:

```bash
conda activate mlops
dvc repro
```

Inspect the stage graph and tracked metrics with:

```bash
dvc dag
dvc metrics show
```

DVC keeps generated data and model outputs under `data/`, `reports/`, and
`artifacts/`, and records their dependencies in `dvc.lock`. The deploy stage
creates `artifacts/iris_model.joblib`, which the FastAPI app loads for inference.

## Run the web app

```bash
conda activate mlops
python -m uvicorn ml_pipeline.api:app --reload
```

Open <http://127.0.0.1:8000> for the interface or <http://127.0.0.1:8000/docs>
for the interactive API documentation. The app trains the model on startup if
the artifact does not exist. The interface can also rerun the pipeline.

### API endpoints

- `GET /health` reports model readiness and the latest evaluation metrics.
- `POST /pipeline/run` reruns ingestion, validation, training, evaluation, and
	artifact creation.
- `POST /predict` accepts the four Iris measurements and returns the predicted
	species and confidence.

Example prediction request:

```json
{
	"sepal_length": 5.1,
	"sepal_width": 3.5,
	"petal_length": 1.4,
	"petal_width": 0.2
}
```
