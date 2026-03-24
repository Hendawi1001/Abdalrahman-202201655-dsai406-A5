import os
import sys
import mlflow

def check_model_performance():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:///tmp/mlruns"))

    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        print("Error: model_info.txt not found.")
        sys.exit(1)

    print(f"Checking metrics for Run ID: {run_id}")

    try:
        run = mlflow.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy", 0.0) 
    except Exception as e:
        print(f"Failed to fetch run from MLflow: {e}")
        sys.exit(1)

    threshold = 0.85
    print(f"Model Accuracy: {accuracy:.4f} (Threshold: {threshold})")

    if accuracy >= threshold:
        print("✅ Model passed the validation threshold.")
        sys.exit(0) # Success
    else:
        print("❌ Model failed to meet the accuracy threshold. Stopping deployment.")
        sys.exit(1) # Failure

if __name__ == "__main__":
    check_model_performance()