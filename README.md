# Telco Customer Churn Prediction

Predicts customer churn using the Telco Customer Churn dataset.

## Setup

```bash
make venv
```

## Usage

Default paths (via Makefile):

```bash
make preprocess
make train
make evaluate
make predict
```

Custom paths (via argparse):

```bash
.venv/bin/python src/preprocess.py --input data/raw/other.csv --output data/processed/out.csv
.venv/bin/python src/train.py --data data/processed/out.csv --model-out models/v2.pkl
.venv/bin/python src/evaluate.py --model models/v2.pkl --data data/processed/out.csv
.venv/bin/python src/predict.py --model models/v2.pkl --data data/processed/out.csv
```

## Testing

```bash
make test
```

## Project Structure

```text
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_preprocess.py
│   └── test_train.py
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
