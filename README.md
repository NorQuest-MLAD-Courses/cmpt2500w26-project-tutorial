# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer will churn. The project spans data preprocessing, model training, experiment tracking, hyperparameter tuning, a REST API, containerisation, cloud deployment on Google Cloud Run, and CI/CD with GitHub Actions.


## Project structure

```
churn-prediction/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Run tests on every push and PR
│       └── deploy.yml          # Build, push, and deploy on push to main
├── src/
│   ├── preprocess.py           # Data cleaning and feature encoding
│   ├── train.py                # Model training with MLflow logging
│   ├── tune.py                 # Hyperparameter search (deployment-ready pipelines)
│   ├── evaluate.py             # Model evaluation and metrics
│   ├── predict.py              # CLI predictions from a saved model
│   ├── app.py                  # Flask REST API with structured logging
│   └── utils/
│       ├── __init__.py
│       └── config.py           # YAML config loader
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py
│   ├── test_config.py
│   └── test_api.py
├── config/
│   └── default.yaml            # Centralised project settings
├── models/                     # Saved model artifacts (.pkl, git-ignored)
├── data/
│   ├── raw/                    # Original CSV (tracked by DVC)
│   └── processed/              # Cleaned CSV (tracked by DVC)
├── logs/                       # Application logs (git-ignored)
├── Dockerfile                  # API container image
├── Dockerfile.mlflow           # MLflow server container image
├── docker-compose.yml          # Multi-container orchestration
├── .dockerignore               # Files excluded from Docker build context
├── requirements.txt            # Python dependencies
├── Makefile                    # Automation targets
├── pytest.ini                  # Test configuration
├── .gitignore
└── README.md
```


## Quick start

### Prerequisites

Python 3.10+, pip, and Git. For containerised deployment, Docker and Docker Compose. For cloud deployment, the Google Cloud CLI (`gcloud`). GitHub Codespaces includes all of these except `gcloud`, which `make setup` will install automatically.

### Local setup

```bash
git clone <your-repo-url>
cd churn-prediction
make venv
dvc pull
```

### Run the ML pipeline

```bash
make preprocess
make train
make evaluate
```

### Hyperparameter tuning

```bash
make tune
```

Produces `models/model_v1.pkl` (best F1) and `models/model_v2.pkl` (second-best).

### Run the API locally

```bash
make api
```

The API starts on `http://localhost:5000`. Swagger docs at `/apidocs/`.


## Docker

### Single container (API only)

```bash
make docker-build
make docker-run
```

### Docker Compose (API + MLflow)

```bash
make compose-up
```

| Service | Port | URL |
|---------|------|-----|
| API | 5000 | `http://localhost:5000` |
| MLflow UI | 5001 | `http://localhost:5001` |

Run training inside the container (logs to the MLflow container):

```bash
make compose-train
make compose-tune
```

Stop:

```bash
make compose-down
```


## Cloud deployment

The API is deployed to Google Cloud Run as a serverless container. Cloud Run scales to zero when idle and starts on demand, keeping costs at $0 for low-traffic usage.

### Manual deployment

```bash
DOCKER_USER=<your-dockerhub-username> make docker-push
DOCKER_USER=<your-dockerhub-username> make deploy
```

This pushes the image to Docker Hub and deploys it to Cloud Run. The command outputs a public HTTPS URL like `https://churn-api-xxxxx-uc.a.run.app`.

### CI/CD with GitHub Actions

Every push to `main` and every pull request triggers the CI workflow (`.github/workflows/ci.yml`), which runs the full pytest suite.

Pushes to `main` additionally trigger the CD workflow (`.github/workflows/deploy.yml`), which builds the Docker image, pushes it to Google Artifact Registry, and deploys it to Cloud Run. This requires two GitHub repository secrets:

- `GCP_SA_KEY` — a JSON service account key with Cloud Run and Artifact Registry permissions.
- `GCP_PROJECT_ID` — your Google Cloud project ID.

After setup, every merged pull request automatically deploys the latest code to production.


## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Returns `{"status": "ok"}` if the service is running |
| GET | `/info` | Returns available endpoints, required features, and an example payload |
| POST | `/v1/predict` | Churn prediction using model v1 (best F1) |
| POST | `/v2/predict` | Churn prediction using model v2 (second-best F1) |
| GET | `/apidocs/` | Interactive Swagger UI for testing endpoints |

### Example request

```bash
curl -X POST https://<your-cloud-run-url>/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12, "MonthlyCharges": 59.95, "TotalCharges": 720.50,
    "Contract": "One year", "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No", "TechSupport": "No", "InternetService": "DSL",
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes",
    "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "PaperlessBilling": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "StreamingTV": "No", "StreamingMovies": "No"
  }'
```


## Logging

The API uses Python's `logging` module. Logs are written to stdout (captured by `docker logs` and Cloud Run's log viewer) and to `logs/api.log` (persisted via volume mount in Docker Compose).


## Makefile reference

| Target | Description |
|--------|-------------|
| `venv` | Create virtual environment and install dependencies |
| `preprocess` | Run data cleaning and feature engineering |
| `train` | Train a model and log to MLflow |
| `tune` | Hyperparameter search; saves top two deployment-ready models |
| `evaluate` | Evaluate saved model on test set |
| `predict` | Run CLI prediction |
| `test` | Run pytest suite |
| `api` | Start Flask API locally (development server) |
| `mlflow-ui` | Launch MLflow tracking UI |
| `dvc-push` / `dvc-pull` | Push/pull data to/from DVC remote |
| `docker-build` | Build the API Docker image |
| `docker-run` / `docker-stop` | Run/stop the API container |
| `compose-up` / `compose-down` | Start/stop API + MLflow via Docker Compose |
| `compose-train` / `compose-tune` | Run training/tuning inside the container |
| `docker-tag` / `docker-push` | Tag and push images to Docker Hub |
| `deploy` | Deploy API image to Google Cloud Run |
| `clean` | Remove generated files |


## Technologies

Python, pandas, NumPy, scikit-learn, MLflow, DVC, Flask, Flasgger, Gunicorn, Docker, Docker Compose, Google Cloud Run, GitHub Actions, pytest, PyYAML.
