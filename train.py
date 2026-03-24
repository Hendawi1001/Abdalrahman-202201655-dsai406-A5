import mlflow
import os

SIMULATED_ACCURACY = 0.82 

if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:///tmp/mlruns"))
    
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", SIMULATED_ACCURACY)
        run_id = run.info.run_id
        
    with open("model_info.txt", "w") as f:
        f.write(run_id)
        
    print(f"Training complete. Logged accuracy: {SIMULATED_ACCURACY}")
    print(f"Run ID {run_id} saved to model_info.txt")