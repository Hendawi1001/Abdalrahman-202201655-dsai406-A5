FROM python:3.10-slim
ARG RUN_ID
ENV MLFLOW_RUN_ID=${RUN_ID}

WORKDIR /app

RUN pip install --no-cache-dir mlflow

RUN echo "Downloading model artifacts for MLflow Run ID: ${MLFLOW_RUN_ID}..."

COPY . .

CMD ["echo", "Model container is ready!"]