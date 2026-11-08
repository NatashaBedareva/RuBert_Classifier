.PHONY: install install-dev test lint format train train-quick train-full predict predict-cli sweep clean help

# --- Установка ---
install:
	uv sync

install-dev:
	uv sync --extra dev
	uv run pre-commit install

# --- Качество кода ---
test:
	uv run python -m pytest tests/ -v

lint:
	uv run flake8 src/rubert_classifier tests/
	uv run mypy src/rubert_classifier

format:
	uv run black src/ tests/ run_train.py run_predict.py
	uv run isort src/ tests/ run_train.py run_predict.py

# --- Обучение ---
train:
	uv run python run_train.py

train-quick:
	uv run python run_train.py training=quick preprocessing.target_samples_per_class=10

train-full:
	uv run python run_train.py training=full

# --- Инференс ---
predict:
	uv run python run_predict.py

predict-cli:
	uv run rubert-clf predict --model-dir ./best_model

# --- Эксперименты (Hydra Multirun) ---
sweep:
	uv run python run_train.py -m training=quick preprocessing.target_samples_per_class=10 training.learning_rate=1e-5,3e-5,5e-5 best_model_dir=./best_models/lr_$${training.learning_rate}

# --- Очистка ---
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov/ .coverage
	rm -rf outputs/ multirun/
	find . -type d -name __pycache__ -exec rm -rf {} +