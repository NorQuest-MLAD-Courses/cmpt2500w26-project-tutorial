.PHONY: venv preprocess train tune evaluate predict test api mlflow-ui dvc-push dvc-pull docker-build docker-run docker-stop compose-up compose-down compose-train compose-tune docker-tag docker-push clean

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

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
	docker compose run --rm --workdir /app/src api python train.py

compose-tune:
	docker compose run --rm --workdir /app/src api python tune.py

docker-tag:
	docker tag churn-api $(DOCKER_USER)/churn-api:latest
	docker tag $(shell docker compose images mlflow -q) $(DOCKER_USER)/churn-mlflow:latest

docker-push:
	docker push $(DOCKER_USER)/churn-api:latest
	docker push $(DOCKER_USER)/churn-mlflow:latest

clean:
	rm -rf .venv models/*.pkl data/processed/*.csv mlruns/ logs/
