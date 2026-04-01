"""
drift.py — Generate a data drift report comparing reference and current data.
Uses Evidently to detect changes in feature distributions that might
indicate the model is seeing data it was not trained on.
"""

import argparse
import os
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from utils.config import load_config
from preprocess import load_and_clean


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a data drift report.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--current", default=None,
                        help="Path to current (production) data CSV. "
                             "If not provided, the script simulates drift "
                             "by perturbing the reference data.")
    parser.add_argument("--output", default="reports/drift_report.html",
                        help="Path to save the HTML report.")
    return parser.parse_args()


def simulate_drift(df, random_state=42):
    """
    Create a 'drifted' copy of the data by perturbing some features.
    This simulates what happens when production data diverges from
    the training distribution.
    """
    rng = np.random.RandomState(random_state)
    drifted = df.copy()

    # Shift numerical features
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] * 1.3 + rng.normal(0, 10, len(drifted))
    drifted["tenure"] = (drifted["tenure"] * 0.7).clip(lower=1).astype(int)

    # Change categorical distribution: more month-to-month contracts
    mask = rng.random(len(drifted)) < 0.4
    drifted.loc[mask, "Contract"] = "Month-to-month"

    return drifted


def main():
    args = parse_args()
    cfg = load_config(args.config)

    raw_path = cfg["paths"]["raw_data"]
    print(f"Loading reference data from {raw_path} ...")
    reference = load_and_clean(raw_path)

    if "Churn" in reference.columns:
        reference = reference.drop(columns=["Churn"])

    if args.current:
        print(f"Loading current data from {args.current} ...")
        current = pd.read_csv(args.current)
    else:
        print("No current data provided — simulating drift ...")
        current = simulate_drift(reference)

    if "Churn" in current.columns:
        current = current.drop(columns=["Churn"])

    print("Generating drift report ...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
    ])

    snapshot = report.run(reference_data=reference, current_data=current)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    snapshot.save_html(args.output)
    print(f"Report saved to {args.output}")
    print("Open it in your browser to explore the results.")


if __name__ == "__main__":
    main()
