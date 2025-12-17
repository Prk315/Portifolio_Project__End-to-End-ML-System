.PHONY: help install train-baseline train-advanced api ui docker-up docker-down test clean

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make train-baseline  - Train baseline model"
	@echo "  make train-advanced  - Train advanced model"
	@echo "  make api            - Start API server"
	@echo "  make ui             - Start Streamlit UI"
	@echo "  make docker-up      - Start Docker services"
	@echo "  make docker-down    - Stop Docker services"
	@echo "  make test           - Run tests"
	@echo "  make clean          - Clean generated files"

install:
	pip install -r requirements.txt

train-baseline:
	python src/models/train_baseline.py --data data/raw/accepted_2007_to_2018Q4.csv

train-advanced:
	python src/models/train_advanced.py --data data/raw/accepted_2007_to_2018Q4.csv

train-advanced-tune:
	python src/models/train_advanced.py --data data/raw/accepted_2007_to_2018Q4.csv --tune

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/app.py

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

test:
	pytest tests/ -v --cov=src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -f .coverage
