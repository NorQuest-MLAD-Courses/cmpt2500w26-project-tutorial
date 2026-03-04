# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer will churn (cancel their service). The trained models are served through a REST API, enabling any application to request churn predictions over HTTP.

## Problem Description

Customer churn — the loss of clients to a competitor or cancellation of service — is a major revenue concern for telecom providers. This project trains binary classifiers on historical customer data, tracks experiments with MLflow, versions data with DVC, and exposes the best models through a Flask REST API with interactive Swagger documentation.

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
│   ├── model_v1.pkl            # Best model (deployment-ready pipeline)
│   └── model_v2.pkl            # Second-best model (deployment-ready pipeline)
├── src/
│   ├── app.py                  # Flask REST API (prediction service)
│   ├── preprocess.py           # Load raw data, clean, encode, save
│   ├── train.py                # Train pipeline, log to MLflow, save model
│   ├── tune.py                 # Hyperparameter search across model families
│   ├── evaluate.py             # Load model, report metrics on test split
│   ├── predict.py              # Load model, output predictions (CLI)
│   └── utils/
│       ├── __init__.py
│       └── config.py           # YAML config loader
├── tests/
│   ├── test_api.py             # API endpoint tests (Flask test client)
│   ├── test_preprocess.py      # Data cleaning and encoding tests
│   ├── test_train.py           # Pipeline construction and fitting tests
│   └── test_config.py          # Config loading tests
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

### Running the API

The primary interface to the models is the REST API:

```bash
make api
# or: python src/app.py
```

The server starts on `http://127.0.0.1:5000`. In GitHub Codespaces, click "Make Public" when prompted and use the forwarded URL.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/info` | API documentation — lists endpoints, features, and a request example |
| `POST` | `/v1/predict` | Churn prediction using model v1 (best model) |
| `POST` | `/v2/predict` | Churn prediction using model v2 (second-best model) |
| `GET` | `/apidocs/` | Interactive Swagger UI with live request testing |

### Making Predictions

Send a `POST` request with customer data as JSON:

```bash
curl -X POST http://127.0.0.1:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 59.95,
    "TotalCharges": 720.50,
    "Contract": "One year",
    "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "InternetService": "DSL",
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "PaperlessBilling": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No"
  }'
```

Response:

```json
{
  "prediction": "No",
  "probability": 0.9431,
  "model_version": "v1"
}
```

Batch predictions are supported — send a JSON array to predict multiple customers in one request.

### ML Pipeline (CLI)

The training and evaluation pipeline is still available via the command line:

```bash
make preprocess    # Clean raw data → data/processed/churn_processed.csv
make train         # Train model   → models/model.pkl (+ MLflow logging)
make tune          # Hyperparameter search → models/model_v1.pkl, model_v2.pkl
make evaluate      # Report metrics on the held-out test split
make predict       # Output churn predictions for each row
```

### Running Tests

```bash
make test                          # Run all tests
pytest tests/test_api.py -v        # Run API tests only
```

### Viewing Experiment History

```bash
make mlflow-ui
# Open http://127.0.0.1:5000 in your browser
# (stop the API server first, or run MLflow on a different port)
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

## Architecture

### Training Pipeline

1. **Preprocessing** (`src/preprocess.py`): Loads the raw CSV, drops `customerID`, converts `TotalCharges` to numeric, removes zero-tenure rows, maps `SeniorCitizen` from 0/1 to "No"/"Yes", label-encodes all categorical columns, and writes the result as a processed CSV.

2. **Training** (`src/train.py`): Reads the processed CSV, splits into train/test sets (stratified), builds a scikit-learn `Pipeline` with `StandardScaler` on numerical features and a `GradientBoostingClassifier`, fits it, evaluates on the test set, logs parameters and metrics to MLflow, and saves the pipeline as a pickle.

3. **Hyperparameter Tuning** (`src/tune.py`): Searches across model families (GradientBoosting, RandomForest, LogisticRegression) and hyperparameter grids. Builds deployment-ready pipelines that include `OrdinalEncoder` for categoricals, so saved models accept raw string data directly. Logs all runs to MLflow and saves the top two by F1 score as `model_v1.pkl` and `model_v2.pkl`.

### Serving Pipeline

4. **REST API** (`src/app.py`): A Flask application that loads `model_v1` and `model_v2` at startup and exposes prediction endpoints. Accepts raw customer data as JSON, validates input, runs the model, and returns predictions. Flasgger provides interactive Swagger documentation at `/apidocs/`.

### Model Artifacts

The project maintains two types of model:

- **`models/model.pkl`** — The standard training pipeline (from `train.py`), which expects pre-encoded data. Used for CLI evaluation and prediction.
- **`models/model_v1.pkl` / `model_v2.pkl`** — Deployment-ready pipelines (from `tune.py`) with `OrdinalEncoder` built in. Accept raw string categorical data. Used by the API.

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

MLflow records parameters, metrics, and model artifacts for every training run. The tuning script creates a `churn-tuning` experiment with all hyperparameter combinations for easy comparison.

```bash
make mlflow-ui
# Select the "churn-tuning" experiment
# Sort by f1_score to see the best models
```

## Technologies

- **Python 3.12** — Language
- **Flask** — REST API framework
- **Flasgger** — Swagger/OpenAPI documentation
- **scikit-learn** — ML pipelines, preprocessing, classification
- **pandas / NumPy** — Data manipulation
- **MLflow** — Experiment tracking and model logging
- **DVC** — Data version control (DagsHub remote)
- **pytest** — Automated testing (including API endpoint tests)
- **PyYAML** — Configuration management
- **Make** — Task automation

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 rows, 21 columns. Each row represents a customer; the target column is `Churn` (Yes/No).
