"""
tune.py — Hyperparameter search across model families.
Produces deployment-ready pipelines that handle full preprocessing internally.
"""

import argparse
import pickle
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from utils.config import load_config
from preprocess import load_and_clean


# --- Column definitions ---
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def build_deployment_pipeline(classifier, num_cols, cat_cols):
    """Build a pipeline that accepts cleaned (not encoded) data."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), cat_cols),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# --- Candidate models and grids ---
CANDIDATES = {
    "GradientBoosting": {
        "class": GradientBoostingClassifier,
        "grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
        },
    },
    "LogisticRegression": {
        "class": LogisticRegression,
        "grid": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "max_iter": [1000],
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter search.")
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("churn-tuning")

    # Load raw data and apply cleaning only (no encoding)
    raw_path = cfg["paths"]["raw_data"]
    df = load_and_clean(raw_path)

    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )

    results = []

    for model_name, spec in CANDIDATES.items():
        print(f"\n--- {model_name} ---")
        grid = spec["grid"]

        # Generate all combinations
        from itertools import product as cartesian_product
        keys = list(grid.keys())
        for combo in cartesian_product(*grid.values()):
            params = dict(zip(keys, combo))
            classifier = spec["class"](**params, random_state=cfg["training"]["random_state"])
            pipeline = build_deployment_pipeline(classifier, NUM_COLS, CAT_COLS)

            with mlflow.start_run(run_name=f"{model_name}_{params}"):
                mlflow.log_param("model_family", model_name)
                for k, v in params.items():
                    mlflow.log_param(k, v)

                pipeline.fit(X_train, y_train)
                yhat = pipeline.predict(X_test)

                acc = accuracy_score(y_test, yhat)
                f1 = f1_score(y_test, yhat)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.sklearn.log_model(pipeline, "model")

                print(f"  {params} → acc={acc:.4f}  f1={f1:.4f}")
                results.append({
                    "model_name": model_name,
                    "params": params,
                    "accuracy": acc,
                    "f1_score": f1,
                    "pipeline": pipeline,
                })

    # Rank by F1 and save top two
    results.sort(key=lambda r: r["f1_score"], reverse=True)

    for rank, label in [(0, "v1"), (1, "v2")]:
        best = results[rank]
        out_path = f"models/model_{label}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(best["pipeline"], f)
        print(f"\nSaved {label}: {best['model_name']} "
              f"(f1={best['f1_score']:.4f}) → {out_path}")


if __name__ == "__main__":
    main()
