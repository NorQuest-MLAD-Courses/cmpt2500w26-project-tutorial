# Telco Customer Churn Prediction

Predicts customer churn using the Telco Customer Churn dataset.

## Setup

```bash
make venv
```

## Usage

```bash
make preprocess   # clean and encode raw data
make train        # train pipeline and save to models/
make evaluate     # evaluate saved pipeline on test split
make predict      # print predictions to screen
make test         # run test suite
```

## Project Structure

```
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
