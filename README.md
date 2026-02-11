# Telco Customer Churn Prediction

Predicts customer churn using the Telco Customer Churn dataset.

## Setup

```bash
make venv
```

## Usage

All settings are read from `config/default.yaml`. Command-line flags can override paths.

```bash
make preprocess
make train
make evaluate
make predict
```

Use a custom config:

```bash
.venv/bin/python src/train.py --config config/experiment.yaml
```

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
├── pytest.ini
├── Makefile
├── requirements.txt
└── README.md
```
