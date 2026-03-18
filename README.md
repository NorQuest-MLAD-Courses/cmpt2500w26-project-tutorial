# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer will churn. The project spans data preprocessing, model training, experiment tracking, hyperparameter tuning, a REST API for real-time predictions, containerisation with Docker, and publishing to Docker Hub.


## Project structure

```
churn-prediction/
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
│   │   └── raw.dvc             # DVC metadata pointer
│   └── processed/              # Cleaned CSV (tracked by DVC)
│       └── processed.dvc       # DVC metadata pointer
├── logs/                       # Application logs (git-ignored)
├── .dvc/
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

Python 3.10+, pip, and Git. For containerised deployment, Docker and Docker Compose are also required. GitHub Codespaces environments include all of these out of the box.

### Local setup

```bash
git clone <your-repo-url>
cd churn-prediction
make venv
```

Pull the data from the DVC remote (requires credentials configured once):

```bash
dvc remote modify origin --local access_key_id <YOUR_TOKEN>
dvc remote modify origin --local secret_access_key <YOUR_TOKEN>
dvc pull
```

### Run the ML pipeline

```bash
make preprocess
make train
make evaluate
```

### Hyperparameter tuning

The tuning script searches across GradientBoosting, RandomForest, and LogisticRegression with full parameter grids. Each run is logged to MLflow. The top two models are saved as deployment-ready pipelines that handle all preprocessing internally.

```bash
make tune
```

This produces `models/model_v1.pkl` (best F1) and `models/model_v2.pkl` (second-best F1). These files are required before running the API or building the Docker image.

### Run the API locally

```bash
make api
```

The API starts on `http://localhost:5000`. Interactive Swagger documentation is available at `/apidocs/`.


## Docker

### Single container (API only)

Build and run the API in a Docker container:

```bash
make docker-build
make docker-run
```

The API serves at `http://localhost:5000`. In Codespaces, use the forwarded URL shown in the Ports tab.

Stop the container:

```bash
make docker-stop
```

### Docker Compose (API + MLflow)

Launch both the prediction API and the MLflow tracking UI as a multi-container application:

```bash
make compose-up
```

This starts two services connected by an internal Docker network (`ml-network`):

| Service | Port | URL |
|---------|------|-----|
| API | 5000 | `http://localhost:5000` |
| MLflow UI | 5001 | `http://localhost:5001` |

In Codespaces, both ports appear in the **Ports** tab. Click the globe icon to open each in your browser.

**Volume mounts.** The Compose setup mounts three directories from the host: `models/` (so the API serves whichever models are on disk), `data/` (so training can run inside the container), and `logs/` (so application logs persist across container restarts). The MLflow container mounts `mlruns/` so experiment history is shared between the container and local tools.

**Networking.** Both services are attached to `ml-network`. Inside the network, the API container can reach the MLflow server at `http://mlflow:5001`. The `MLFLOW_TRACKING_URI` environment variable is set automatically by Compose. This means `train.py` and `tune.py` log directly to the containerised MLflow server when run inside the API container.

**Run training inside the container:**

```bash
make compose-train
make compose-tune
```

These commands execute the training or tuning script inside a temporary API container. Because `MLFLOW_TRACKING_URI` points to the MLflow container, experiment runs are logged there and immediately visible in the MLflow UI at `http://localhost:5001`.

Stop and remove the containers:

```bash
make compose-down
```

### Docker Hub

After verifying everything works, publish your images to Docker Hub:

```bash
docker login
make docker-build
make compose-up          # builds both images
DOCKER_USER=<your-username> make docker-tag
DOCKER_USER=<your-username> make docker-push
```


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
curl -X POST http://localhost:5000/v1/predict \
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

Both endpoints accept a single JSON object or a JSON array for batch predictions.


## Logging

The API uses Python's `logging` module for structured log output. Log messages include timestamps, severity levels, and the module name. Logs are written to both stdout (captured by `docker logs`) and a persistent file at `logs/api.log` (shared with the host via a volume mount).

Predictions, validation failures, and errors are all logged. To view live container logs:

```bash
docker compose logs -f api
```

To view the persistent log file:

```bash
cat logs/api.log
```


## Model artifacts

The project produces two kinds of model:

**Standard pipeline** (`models/model.pkl`) — created by `make train`. Uses the preprocessing from `preprocess.py` (LabelEncoder applied outside the pipeline). Requires the caller to encode categoricals first.

**Deployment-ready pipeline** (`models/model_v1.pkl`, `models/model_v2.pkl`) — created by `make tune`. Has an OrdinalEncoder built into the pipeline's ColumnTransformer alongside StandardScaler. Accepts cleaned but unencoded data directly. These are what the API serves.


## Experiment tracking

MLflow logs every training and tuning run. To browse experiments locally:

```bash
make mlflow-ui
```

When running Docker Compose, the MLflow service runs on port 5001 and the `mlruns/` directory is mounted as a volume. Training runs executed inside the container (via `make compose-train` or `make compose-tune`) log directly to the containerised MLflow server over the internal Docker network.


## Data versioning

Data is managed with DVC and stored on DagsHub (S3-compatible remote). Git tracks `.dvc` metadata files; DVC tracks the actual data.

```bash
dvc add data/processed
git add data/processed.dvc
git commit -m "Update processed data"
dvc push
```

To sync on another machine: `git pull && dvc pull`.


## Testing

Run the full test suite:

```bash
make test
```

Tests use the Flask test client and do not require a running server or Docker container. The suite covers preprocessing logic, training output, YAML config loading, and all API endpoints including validation, batch predictions, and error handling.


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
| `dvc-push` | Push data to DVC remote |
| `dvc-pull` | Pull data from DVC remote |
| `docker-build` | Build the API Docker image |
| `docker-run` | Run the API container (detached, port 5000) |
| `docker-stop` | Stop and remove the API container |
| `compose-up` | Start API + MLflow via Docker Compose |
| `compose-down` | Stop and remove all Compose containers |
| `compose-train` | Run training inside the API container (logs to MLflow container) |
| `compose-tune` | Run tuning inside the API container (logs to MLflow container) |
| `docker-tag` | Tag images for Docker Hub (requires `DOCKER_USER`) |
| `docker-push` | Push tagged images to Docker Hub (requires `DOCKER_USER`) |
| `clean` | Remove virtual environment, models, processed data, mlruns, and logs |


## Technologies

Python, pandas, NumPy, scikit-learn, MLflow, DVC, Flask, Flasgger, Gunicorn, Docker, Docker Compose, pytest, PyYAML.
