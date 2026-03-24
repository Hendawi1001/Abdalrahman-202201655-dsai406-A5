import mlflow
import os
import sys

THRESHOLD = 0.85

if __name__ == "__main__":
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    print(f"Checking Run ID: {run_id}")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:///tmp/mlruns"))
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print("ERROR: 'accuracy' metric not found in MLflow run.")
        sys.exit(1)

    print(f"Accuracy from MLflow: {accuracy}")
    print(f"Threshold:            {THRESHOLD}")

    if accuracy < THRESHOLD:
        print(f"FAILED: Accuracy {accuracy} is below threshold {THRESHOLD}. Blocking deployment.")
        sys.exit(1)

    print(f"PASSED: Accuracy {accuracy} meets the threshold. Proceeding to deploy.")