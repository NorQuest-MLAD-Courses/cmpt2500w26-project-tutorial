"""
train.py — Load processed data, split, train a pipeline, save it, and log to MLflow.
"""

import argparse
import pickle
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from utils.config import load_config


def build_pipeline(num_cols, model_params):
    """Construct a pipeline: scale numerics, pass-through the rest, then classify."""
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_cols)],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(**model_params)),
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Train a churn-prediction pipeline.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data", default=None, help="Override processed data path")
    parser.add_argument("--model-out", default=None, help="Override model output path")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_path = args.data or cfg["paths"]["processed_data"]
    model_path = args.model_out or cfg["paths"]["model"]
    num_cols = cfg["features"]["numerical"]
    test_size = cfg["training"]["test_size"]
    random_state = cfg["training"]["random_state"]
    model_params = cfg["model"]["params"]

    # --- MLflow setup ---
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model", cfg["model"]["name"])
        for k, v in model_params.items():
            mlflow.log_param(f"model_{k}", v)

        # Train
        print("Training pipeline ...")
        pipeline = build_pipeline(num_cols, model_params)
        pipeline.fit(X_train, y_train)

        # Evaluate
        yhat = pipeline.predict(X_test)
        acc = accuracy_score(y_test, yhat)
        f1 = f1_score(y_test, yhat)
        print(f"Test accuracy: {acc:.4f}")
        print(f"Test F1:       {f1:.4f}")
        print(classification_report(y_test, yhat))

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log model artifact
        mlflow.sklearn.log_model(pipeline, "model")

        # Also save locally
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"Pipeline saved to {model_path}")
        print(f"MLflow run logged to {cfg['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
