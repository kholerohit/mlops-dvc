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
- **Experiment tracking:** log training parameters, held-out metrics, and the
	fitted model to the `iris-classification` MLflow experiment on DagsHub.
- **Deployment:** save the fitted pipeline and inference metadata to
	`artifacts/iris_model.joblib`.
- **Frontend/API:** serve a small prediction interface and JSON endpoints with
	FastAPI.

## Setup

```bash
python -m pip install -r requirements.txt
```

The MLflow tracking server is configured for
<https://dagshub.com/kholerohit/mlops-dvc.mlflow>. Authenticate with DagsHub
before running the pipeline; use `DAGSHUB_USER_TOKEN` or the DagsHub CLI login
so runs and model artifacts can be uploaded to the repository.

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

## GitHub Actions CI

The workflow at `.github/workflows/ml-pipeline.yml` runs the test suite on every
push and pull request targeting `main`. After tests pass on a push to `main` (for
example, after a pull request is merged), it runs `demo.py` and logs the trained
model, parameters, and metrics to the DagsHub `iris-classification` experiment.
Pull requests do not receive the DagsHub credential and do not upload runs.

Add a repository Actions secret named `DAGSHUB_USER_TOKEN` in GitHub under
**Settings > Secrets and variables > Actions**. Use a DagsHub token with write
access to `kholerohit/mlops-dvc`. Do not put the token in workflow files or commit
it to the repository.

To require approval before merging, configure branch protection for `main` in
**Settings > Branches** (or the repository rulesets page): require a pull
request, set the required number of approving reviews, and require the
`test-and-track` status check. GitHub branch protection controls approval; the
workflow itself only runs the tracking step after the change has reached `main`.

## Run the web app

```bash
conda activate mlops
python -m uvicorn ml_pipeline.api:app --reload
```

Open <http://127.0.0.1:8000> for the interface or <http://127.0.0.1:8000/docs>
for the interactive API documentation. The app trains the model on startup if
the artifact does not exist. The interface can also rerun the pipeline.

## Run with Docker Compose

Docker Compose runs the published FastAPI image behind an unprivileged Nginx
reverse proxy. Only Nginx is published to the host; the API stays on the private
Compose network. A named volume keeps the trained model across container
restarts.

You need Docker with the Compose plugin. To run the stack from the repository,
use `compose.yaml` and `nginx/default.conf`. The app logs experiments to the
DagsHub repository, so it needs a DagsHub access token when it trains the model
for the first time or when the pipeline is rerun.

Create a local `.env` file from the template and edit it to set your token:

```bash
cp .env.example .env
```

Keep `.env` private. It is Git-ignored; do not commit it or share your personal
token. Each person running the project should use a token with access to the
DagsHub repository.

Pull the app image and start both containers:

```bash
docker compose pull
docker compose up -d
```

Open <http://localhost:8080> for the app or
<http://localhost:8080/docs> for the API docs. To change the host port, edit
`APP_PORT` in `.env` before starting the stack.

Useful commands:

```bash
docker compose ps                         # Check service status and health
docker compose logs -f                    # Follow logs from both containers
docker compose logs -f app                # Follow application logs
docker compose restart                    # Restart both services
docker compose pull && docker compose up -d  # Update to the latest published image
docker compose down                       # Stop and remove containers; keep model data
```

The named `model-artifacts` volume survives `docker compose down` and container
updates. To delete the volume and the saved model too, run
`docker compose down --volumes`.

To run the published app image without Nginx or Compose, create `.env` as above
and run:

```bash
docker run -d --name iris-app \
	-p 8000:8000 \
	--env-file .env \
	-v iris-model:/app/artifacts \
	--restart unless-stopped \
	kholerohit/mlops-dvc-app:latest
```

Then open <http://localhost:8000>. Inspect logs with `docker logs -f iris-app`
and stop the container with `docker stop iris-app`.

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
