# Telco Customer Churn Prediction

Predicts customer churn using the Telco Customer Churn dataset.

## Setup

```bash
make venv
```

## Usage

```bash
make preprocess
make train          # trains and logs to MLflow
make evaluate
make predict
```

## Experiment Tracking

Training runs are logged to MLflow. Launch the dashboard:

```bash
make mlflow-ui
```

Then open <http://127.0.0.1:5000> in a browser.

## Testing

```bash
make test
```

## Project Structure

```
├── config/
│   └── default.yaml
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_config.py
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
├── models/
├── mlruns/              (created by MLflow, git-ignored)
├── pytest.ini
├── Makefile
├── requirements.txt
└── README.md
```
