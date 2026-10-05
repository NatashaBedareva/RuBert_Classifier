.PHONY: install install-dev test lint format train predict clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/

lint:
	flake8 src/rubert_classifier tests/
	mypy src/rubert_classifier

format:
	black src/ tests/
	isort src/ tests/

train:
	rubert-clf train --output-dir ./output --best-model-dir ./best_model --epochs 20

predict:
	rubert-clf predict --model-dir ./best_model

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +