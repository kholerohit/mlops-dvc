from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from ml_pipeline.config import MODEL_PATH
from ml_pipeline.deployment import load_model
from ml_pipeline.orchestrator import run_pipeline


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sepal_length: float = Field(gt=0, description="Sepal length in centimeters")
    sepal_width: float = Field(gt=0, description="Sepal width in centimeters")
    petal_length: float = Field(gt=0, description="Petal length in centimeters")
    petal_width: float = Field(gt=0, description="Petal width in centimeters")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists():
        run_pipeline()
    yield


app = FastAPI(
    title="Iris ML Pipeline Demo",
    description="A local, modular machine-learning workflow and prediction service.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Iris Model Console</title>
  <style>
    :root { color-scheme: light; font-family: ui-sans-serif, system-ui, sans-serif; color: #17231f; background: #f3f5ef; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: linear-gradient(135deg, #f3f5ef 0%, #e7f0ec 58%, #f5eee2 100%); }
    main { width: min(920px, calc(100% - 32px)); margin: 48px auto; }
    header { display: flex; justify-content: space-between; align-items: end; gap: 24px; border-bottom: 1px solid #becbc4; padding-bottom: 22px; }
    .eyebrow { color: #28725d; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    h1 { margin: 8px 0 0; font-family: Georgia, serif; font-size: clamp(32px, 6vw, 54px); font-weight: 500; }
    #status { color: #53645c; font-size: 14px; text-align: right; }
    section { padding: 28px 0; border-bottom: 1px solid #becbc4; }
    h2 { margin: 0 0 18px; font-size: 18px; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #cbd5cf; border: 1px solid #cbd5cf; }
    .metric { padding: 16px; background: #f8faf6; }
    .metric span { display: block; color: #58685f; font-size: 12px; }
    .metric strong { display: block; margin-top: 7px; font-size: 22px; font-variant-numeric: tabular-nums; }
    form { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    label { display: grid; gap: 7px; color: #53645c; font-size: 13px; }
    input { width: 100%; border: 1px solid #aebdb5; border-radius: 4px; padding: 11px; background: #fff; color: #17231f; font: inherit; }
    button { min-height: 42px; padding: 0 16px; border: 0; border-radius: 4px; background: #176b54; color: white; font: inherit; font-weight: 650; cursor: pointer; }
    button:hover { background: #105640; }
    .actions { display: flex; align-items: center; gap: 16px; margin-top: 18px; }
    #result { color: #176b54; font-weight: 700; }
    #error { color: #a43c31; }
    @media (max-width: 650px) { main { margin: 28px auto; } header { align-items: start; flex-direction: column; } #status { text-align: left; } .metrics { grid-template-columns: repeat(2, 1fr); } form { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
  <main>
    <header>
      <div><div class="eyebrow">Machine learning / local demo</div><h1>Iris model console</h1></div>
      <div id="status" role="status">Checking model service...</div>
    </header>
    <section>
      <h2>Held-out evaluation</h2>
      <div class="metrics" id="metrics"><div class="metric"><span>Accuracy</span><strong>--</strong></div><div class="metric"><span>Precision</span><strong>--</strong></div><div class="metric"><span>Recall</span><strong>--</strong></div><div class="metric"><span>F1</span><strong>--</strong></div></div>
      <div class="actions"><button type="button" id="run">Run pipeline again</button><span id="pipeline-message"></span></div>
    </section>
    <section>
      <h2>Classify a flower</h2>
      <form id="prediction-form">
        <label>Sepal length (cm)<input name="sepal_length" type="number" min="0.1" step="0.1" value="5.1" required></label>
        <label>Sepal width (cm)<input name="sepal_width" type="number" min="0.1" step="0.1" value="3.5" required></label>
        <label>Petal length (cm)<input name="petal_length" type="number" min="0.1" step="0.1" value="1.4" required></label>
        <label>Petal width (cm)<input name="petal_width" type="number" min="0.1" step="0.1" value="0.2" required></label>
      </form>
      <div class="actions"><button type="submit" form="prediction-form">Predict species</button><span id="result" aria-live="polite"></span><span id="error" aria-live="polite"></span></div>
    </section>
  </main>
  <script>
    const statusNode = document.querySelector('#status');
    const metricsNode = document.querySelector('#metrics');
    async function refreshStatus() {
      const response = await fetch('/health');
      const data = await response.json();
      statusNode.textContent = data.model_loaded ? 'Model ready · ' + data.dataset : 'Model not available';
      if (data.metrics) {
        const names = [['Accuracy', 'accuracy'], ['Precision', 'precision_macro'], ['Recall', 'recall_macro'], ['F1', 'f1_macro']];
        metricsNode.innerHTML = names.map(([label, key]) => `<div class="metric"><span>${label}</span><strong>${(data.metrics[key] * 100).toFixed(1)}%</strong></div>`).join('');
      }
    }
    document.querySelector('#run').addEventListener('click', async () => {
      const message = document.querySelector('#pipeline-message');
      message.textContent = 'Running...';
      const response = await fetch('/pipeline/run', { method: 'POST' });
      const data = await response.json();
      message.textContent = response.ok ? `Complete · ${data.data.rows} rows · ${data.split.train_rows} train / ${data.split.test_rows} test` : data.detail;
      await refreshStatus();
    });
    document.querySelector('#prediction-form').addEventListener('submit', async (event) => {
      event.preventDefault();
      document.querySelector('#error').textContent = '';
      const formData = new FormData(event.currentTarget);
      const payload = Object.fromEntries([...formData].map(([key, value]) => [key, Number(value)]));
      const response = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const data = await response.json();
      if (!response.ok) document.querySelector('#error').textContent = data.detail;
      else document.querySelector('#result').textContent = `${data.species} · confidence ${(data.confidence * 100).toFixed(1)}%`;
    });
    refreshStatus().catch(() => { statusNode.textContent = 'Could not connect to model service'; });
  </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {"status": "not_ready", "model_loaded": False}
    artifact = load_model(MODEL_PATH)
    return {
        "status": "ready",
        "model_loaded": True,
        "dataset": artifact["description"],
        "metrics": artifact["metrics"],
    }


@app.post("/pipeline/run")
def pipeline_run() -> dict[str, Any]:
    return run_pipeline()


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        artifact = load_model(MODEL_PATH)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Model is not ready. Run the pipeline first.") from error

    values = {
        "sepal length (cm)": request.sepal_length,
        "sepal width (cm)": request.sepal_width,
        "petal length (cm)": request.petal_length,
        "petal width (cm)": request.petal_width,
    }
    features = pd.DataFrame([values], columns=artifact["feature_names"])
    model = artifact["model"]
    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    return {
        "class_id": predicted_class,
        "species": artifact["target_names"][predicted_class],
        "confidence": float(probabilities[predicted_class]),
    }