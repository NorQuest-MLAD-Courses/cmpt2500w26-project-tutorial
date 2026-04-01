SHELL := /bin/bash

.PHONY: setup venv preprocess train tune evaluate predict test api mlflow-ui dvc-push dvc-pull docker-build docker-run docker-stop compose-up compose-down compose-train compose-tune docker-tag docker-push deploy drift compose-drift clean

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

setup:
	@echo "Creating virtual environment ..."
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Configuring DVC remote credentials ..."
	@read -s -p "DagsHub Access Key ID: " KEY_ID && echo "" && \
	 read -s -p "DagsHub Secret Access Key: " SECRET_KEY && echo "" && \
	 .venv/bin/dvc remote modify origin --local access_key_id "$$KEY_ID" && \
	 .venv/bin/dvc remote modify origin --local secret_access_key "$$SECRET_KEY"
	@echo "Pulling data from DVC remote ..."
	.venv/bin/dvc pull
	@echo ""
	@echo "Installing Google Cloud CLI ..."
	@if ! command -v gcloud &> /dev/null; then \
		echo "gcloud not found — installing ..."; \
		curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts; \
		echo ""; \
		echo "gcloud installed. Restart your shell (exec -l $$SHELL) then run:"; \
		echo "  gcloud auth login"; \
		echo "  gcloud config set project <YOUR_PROJECT_ID>"; \
	else \
		echo "gcloud already installed: $$(gcloud --version 2>&1 | head -1)"; \
	fi
	@echo ""
	@echo "Setup complete. Run 'make test' to verify."

preprocess:
	.venv/bin/python src/preprocess.py

train:
	.venv/bin/python src/train.py

tune:
	.venv/bin/python src/tune.py

evaluate:
	.venv/bin/python src/evaluate.py

predict:
	.venv/bin/python src/predict.py

test:
	.venv/bin/pytest

api:
	.venv/bin/python src/app.py

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri mlruns

dvc-push:
	dvc push

dvc-pull:
	dvc pull

docker-build:
	docker build -t churn-api .

docker-run:
	docker run -d --name churn-api -p 5000:5000 churn-api

docker-stop:
	docker stop churn-api && docker rm churn-api

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

compose-train:
	docker compose run --rm api python src/train.py

compose-tune:
	docker compose run --rm api python src/tune.py

docker-tag:
	docker tag churn-api $(DOCKER_USER)/churn-api:latest
	docker tag $(shell docker compose images mlflow -q) $(DOCKER_USER)/churn-mlflow:latest

docker-push:
	docker push $(DOCKER_USER)/churn-api:latest
	docker push $(DOCKER_USER)/churn-mlflow:latest

deploy:
	gcloud run deploy churn-api \
		--image $(DOCKER_USER)/churn-api:latest \
		--platform managed \
		--region us-central1 \
		--port 5000 \
		--allow-unauthenticated

clean:
	rm -rf .venv models/*.pkl data/processed/*.csv mlruns/ logs/ reports/

drift:
	.venv/bin/python src/drift.py

compose-drift:
	docker compose run --rm -v ./reports:/app/reports api python src/drift.py
