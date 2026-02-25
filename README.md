# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer will churn (cancel their service). Built as a structured, reproducible ML pipeline with experiment tracking and data versioning.

## Problem Description

Customer churn — the loss of clients to a competitor or cancellation of service — is a major revenue concern for telecom providers. This project trains a binary classifier on historical customer data to predict which customers are likely to churn, enabling proactive retention efforts.

The model takes customer attributes (tenure, monthly charges, contract type, services subscribed, etc.) and outputs a churn/no-churn prediction.

## Project Structure

```
├── config/
│   └── default.yaml            # Centralised configuration (paths, features, hyperparameters)
├── data/
│   ├── raw/                    # Original dataset (managed by DVC)
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── processed/              # Cleaned and encoded data (managed by DVC)
│   │   └── churn_processed.csv
│   ├── raw.dvc                 # DVC metadata pointer
│   └── processed.dvc           # DVC metadata pointer
├── models/
│   └── model.pkl               # Trained sklearn pipeline
├── src/
│   ├── preprocess.py           # Load raw data, clean, encode, save
│   ├── train.py                # Train pipeline, log to MLflow, save model
│   ├── evaluate.py             # Load model, report metrics on test split
│   ├── predict.py              # Load model, output predictions
│   └── utils/
│       ├── __init__.py
│       └── config.py           # YAML config loader
├── tests/
│   ├── test_preprocess.py      # Tests for data cleaning and encoding
│   ├── test_train.py           # Tests for pipeline construction and fitting
│   └── test_config.py          # Tests for config loading
├── .dvc/
│   └── config                  # DVC remote configuration
├── Makefile                    # Automation targets
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Test runner configuration
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12+
- Git
- A DagsHub account (for DVC remote storage)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd churn-prediction

# Create a virtual environment and install dependencies
make venv

# Activate the environment
source .venv/bin/activate
```

### Pulling Data with DVC

The dataset and processed files are managed by DVC, not stored in Git. After cloning, pull them from the remote:

```bash
# Set DagsHub credentials (first time or new Codespace only)
dvc remote modify origin --local access_key_id <YOUR_TOKEN>
dvc remote modify origin --local secret_access_key <YOUR_TOKEN>

# Pull data
dvc pull
```

## Usage

All scripts are run from the project root. The `Makefile` provides shortcut targets, or you can run scripts directly.

### Full Pipeline

```bash
make preprocess    # Clean raw data → data/processed/churn_processed.csv
make train         # Train model   → models/model.pkl (+ MLflow logging)
make evaluate      # Report metrics on the held-out test split
make predict       # Output churn predictions for each row
```

### Individual Scripts

Each script accepts a `--config` flag to specify an alternative configuration file, and individual flags to override specific settings:

```bash
# Preprocess with a custom input file
python src/preprocess.py --input data/raw/other_data.csv --output data/processed/other.csv

# Train with a different config
python src/train.py --config config/experimental.yaml

# Evaluate a specific model against specific data
python src/evaluate.py --model models/model.pkl --data data/processed/churn_processed.csv
```

### Running Tests

```bash
make test
# or directly:
pytest
```

### Viewing Experiment History

```bash
make mlflow-ui
# Then open http://127.0.0.1:5000 in your browser
```

## Configuration

All settings are centralised in `config/default.yaml`:

```yaml
paths:
  raw_data: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
  processed_data: data/processed/churn_processed.csv
  model: models/model.pkl

features:
  numerical:
    - tenure
    - MonthlyCharges
    - TotalCharges

training:
  test_size: 0.30
  random_state: 40

model:
  name: GradientBoostingClassifier
  params:
    random_state: 40

mlflow:
  tracking_uri: mlruns
  experiment_name: churn-prediction
```

Create alternative YAML files (e.g., `config/experimental.yaml`) to run experiments with different settings without editing code.

## Pipeline Overview

1. **Preprocessing** (`src/preprocess.py`): Loads the raw CSV, drops `customerID`, converts `TotalCharges` to numeric, removes zero-tenure rows, maps `SeniorCitizen` from 0/1 to "No"/"Yes", label-encodes all categorical columns, and writes the result as a processed CSV.

2. **Training** (`src/train.py`): Reads the processed CSV, splits into train/test sets (stratified), builds a scikit-learn `Pipeline` with `StandardScaler` on numerical features and a `GradientBoostingClassifier`, fits it, evaluates on the test set, logs parameters and metrics to MLflow, and saves the pipeline as a pickle.

3. **Evaluation** (`src/evaluate.py`): Loads a saved pipeline, recreates the same train/test split, and prints accuracy, a classification report, and a confusion matrix.

4. **Prediction** (`src/predict.py`): Loads a saved pipeline and outputs per-row churn/no-churn predictions.

## Data Versioning

Data is managed with DVC and stored on DagsHub (S3-compatible remote). Git tracks `.dvc` metadata files; DVC tracks the actual data.

```bash
# After changing data
dvc add data/processed
git add data/processed.dvc
git commit -m "Update processed data"
dvc push

# Teammates sync with
git pull && dvc pull
```

## Experiment Tracking

MLflow records parameters, metrics, and model artifacts for every training run. Launch the UI with `make mlflow-ui` to compare runs, sort by accuracy or F1, and retrieve any past model.

## Technologies

- **Python 3.12** — Language
- **scikit-learn** — ML pipelines, preprocessing, classification
- **pandas / NumPy** — Data manipulation
- **MLflow** — Experiment tracking and model logging
- **DVC** — Data version control (DagsHub remote)
- **pytest** — Automated testing
- **PyYAML** — Configuration management
- **Make** — Task automation

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 rows, 21 columns. Each row represents a customer; the target column is `Churn` (Yes/No).
