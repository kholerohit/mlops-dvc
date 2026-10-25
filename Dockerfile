FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && mkdir -p /app/artifacts \
    && chown -R app:app /app

COPY requirements-docker.txt ./requirements-docker.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-docker.txt

COPY --chown=app:app ml_pipeline ./ml_pipeline
COPY --chown=app:app demo.py ./demo.py

USER app

EXPOSE 8000

CMD ["uvicorn", "ml_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]