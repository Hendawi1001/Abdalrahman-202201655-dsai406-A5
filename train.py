import mlflow
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "Iris.csv"

if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:///tmp/mlruns"))

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Species", axis=1)
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)

        run_id = run.info.run_id

    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print(f"Training complete. Accuracy: {accuracy:.4f}")
    print(f"Run ID {run_id} saved to model_info.txt")
