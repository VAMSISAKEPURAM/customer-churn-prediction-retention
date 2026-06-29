.PHONY: install install-dev train predict evaluate serve test lint format clean

# ----- Setup -----
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

# ----- Pipelines -----
train:
	python pipelines/train_pipeline.py --config configs/train_config.yaml

predict:
	python pipelines/predict_pipeline.py --config configs/predict_config.yaml

evaluate:
	python pipelines/evaluate_pipeline.py --config configs/train_config.yaml

# ----- API -----
serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# ----- Tests -----
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# ----- Code Quality -----
lint:
	flake8 src/ pipelines/ api/ tests/ --max-line-length=120
	mypy src/ --ignore-missing-imports

format:
	black src/ pipelines/ api/ tests/ --line-length=120
	isort src/ pipelines/ api/ tests/

# ----- Cleanup -----
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# ----- Docker -----
docker-build:
	docker build -f docker/Dockerfile -t churn-model:latest .

docker-run:
	docker run -p 8000:8000 churn-model:latest
