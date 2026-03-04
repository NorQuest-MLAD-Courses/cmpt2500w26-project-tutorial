.PHONY: venv preprocess train tune evaluate predict test api mlflow-ui dvc-push dvc-pull clean

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

preprocess:
	.venv/bin/python src/preprocess.py

train:
	.venv/bin/python src/train.py

tune:
	cd src && ../.venv/bin/python tune.py

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

clean:
	rm -rf .venv models/*.pkl data/processed/*.csv mlruns/
