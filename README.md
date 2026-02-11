# Telco Customer Churn Prediction

Predicts customer churn using the Telco Customer Churn dataset.

## Setup

```bash
make venv
```

## Usage

```bash
make preprocess   # clean and encode raw data
make train        # train model and save to models/
make evaluate     # evaluate saved model on test split
make predict      # print predictions to screen
```

## Project Structure

```
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
├── models/
├── Makefile
├── requirements.txt
└── README.md
```
