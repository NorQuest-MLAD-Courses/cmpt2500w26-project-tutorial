# Telco Customer Churn Prediction

A production-grade machine learning system that predicts whether a telecom customer will churn. Built as an end-to-end MLOps project covering the full lifecycle from raw data to a monitored, cloud-deployed prediction service.

| Layer | Tools |
|-------|-------|
| Data pipeline | pandas, NumPy, DVC |
| Training and tuning | scikit-learn, MLflow |
| API | Flask, Flasgger (Swagger), Gunicorn |
| Containerisation | Docker, Docker Compose |
| Cloud deployment | Google Cloud Run |
| CI/CD | GitHub Actions |
| Infrastructure monitoring | Prometheus, Grafana |
| Training monitoring | Custom Prometheus metrics, psutil |
| Model monitoring | Evidently |
| Alerting | Prometheus alert rules |
| Testing | pytest (17 tests) |


## Project structure

```
churn-prediction/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Run tests on every push and PR
│       └── deploy.yml              # Build, push, and deploy on push to main
├── src/
│   ├── __init__.py
│   ├── preprocess.py               # Data cleaning and feature encoding
│   ├── train.py                    # Model training with MLflow logging
│   ├── tune.py                     # Hyperparameter search (deployment-ready pipelines)
│   ├── evaluate.py                 # Model evaluation and metrics
│   ├── predict.py                  # CLI predictions from a saved model
│   ├── drift.py                    # Evidently data drift detection
│   ├── app.py                      # Flask REST API with logging, Prometheus metrics
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # YAML config loader
│       └── monitoring.py           # Training process Prometheus metrics
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py          # 4 tests
│   ├── test_train.py               # 1 test
│   ├── test_config.py              # 2 tests
│   └── test_api.py                 # 10 tests (health, predict, validation, batch, info, metrics)
├── config/
│   └── default.yaml
├── monitoring/
│   ├── prometheus.yml              # Prometheus scrape and alert configuration
│   ├── rules/
│   │   └── ml_alerts.yml           # Alert rules (error rate, latency, memory, API down)
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   │   └── prometheus.yml
│       │   └── dashboards/
│       │       └── dashboards.yml
│       └── dashboards/
│           ├── api-dashboard.json      # API performance dashboard
│           └── training-dashboard.json # Training monitoring dashboard
├── models/
├── data/
│   ├── raw/
│   └── processed/
├── logs/
├── reports/
├── Dockerfile
├── Dockerfile.mlflow
├── docker-compose.yml              # 4 services: API, MLflow, Prometheus, Grafana
├── .dockerignore
├── requirements.txt
├── Makefile
├── pytest.ini
├── .gitignore
└── README.md
```


## Quick start

```bash
git clone <your-repo-url>
cd churn-prediction
make setup            # Creates venv, prompts for DagsHub keys, pulls data
make preprocess
make train
make tune
```


## Docker Compose (full stack)

Launch all four services:

```bash
make compose-up
```

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| API | 5000 | `http://localhost:5000` | Prediction endpoints + Swagger |
| MLflow | 5001 | `http://localhost:5001` | Experiment tracking UI |
| Prometheus | 9090 | `http://localhost:9090` | Metrics collection, alerting |
| Grafana | 3000 | `http://localhost:3000` | Monitoring dashboards |

Grafana credentials: `admin` / `admin`.


## Monitoring

### API monitoring (Prometheus + Grafana)

The API exposes `/metrics` with both auto-generated HTTP metrics and custom application metrics (prediction count by model version and status, prediction latency histogram, active model gauge). Prometheus scrapes every 5 seconds. Grafana provides two pre-provisioned dashboards:

**API Performance** — request rate, latency percentiles (p50/p95/p99), error rate, prediction requests by model version.

**Training Monitoring** — training vs validation accuracy, training vs validation F1, CPU usage, memory usage, feature importances, epochs completed.

### Training monitoring

`src/utils/monitoring.py` provides a `TrainingMonitor` class that starts a standalone HTTP server on port 8002, exposing training-specific metrics (accuracy, F1, feature importance) and system resource metrics (CPU, memory via psutil). Prometheus scrapes this endpoint alongside the API.

### Alerting

Prometheus alert rules in `monitoring/rules/ml_alerts.yml` fire on four conditions: high prediction error rate (> 10%), slow response time (p95 > 1s), API down (scrape failures), and high training memory (> 1.5 GB). View active alerts at `http://localhost:9090/alerts`.

### Model monitoring (Evidently)

Detect data drift between training and production distributions:

```bash
make drift                    # Simulated drift → reports/drift_report.html
```


## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/info` | API docs, feature definitions, example payload |
| GET | `/metrics` | Prometheus metrics (auto + custom) |
| POST | `/v1/predict` | Prediction using model v1 |
| POST | `/v2/predict` | Prediction using model v2 |
| GET | `/apidocs/` | Interactive Swagger UI |


## Cloud deployment

Deployed to Google Cloud Run. CI/CD via GitHub Actions. See [GCP Setup Guide](gcp_setup_guide.md).

```bash
DOCKER_USER=<username> make docker-push
DOCKER_USER=<username> make deploy
```


## Makefile reference

| Target | Description |
|--------|-------------|
| `setup` | One-command setup: create venv, configure DagsHub credentials, pull data |
| `venv` | Create virtual environment and install dependencies |
| `preprocess` | Clean raw data |
| `train` | Train model, log to MLflow |
| `tune` | Hyperparameter search, save top two models |
| `evaluate` | Evaluate saved model |
| `predict` | CLI prediction |
| `test` | Run pytest suite (17 tests) |
| `api` | Start Flask dev server |
| `mlflow-ui` | Launch MLflow tracking UI |
| `dvc-push` / `dvc-pull` | Push/pull data |
| `docker-build` / `docker-run` / `docker-stop` | Single container |
| `compose-up` / `compose-down` | Full stack (4 services) |
| `compose-train` / `compose-tune` | Train/tune inside container |
| `docker-tag` / `docker-push` | Push to Docker Hub |
| `deploy` | Deploy to Cloud Run |
| `drift` / `compose-drift` | Generate drift report |
| `clean` | Remove generated files |


## Technologies

Python, pandas, NumPy, scikit-learn, MLflow, DVC, Flask, Flasgger, Gunicorn, Docker, Docker Compose, Prometheus, Grafana, Evidently, psutil, Google Cloud Run, GitHub Actions, pytest, PyYAML.
