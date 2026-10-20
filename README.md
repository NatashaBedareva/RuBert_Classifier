# RuBERT Classifier

Text classification on the [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) dataset (Russian) using `DeepPavlov/rubert-base-cased`.

## Features

- 📦 Installable Python package
- ⚡ Fast dependency management with [uv](https://docs.astral.sh/uv/)
- 🚀 CLI for training and inference
- 🗂️ Data & model versioning with [DVC](https://dvc.org/)
- ⚖️ Automatic class balancing (oversampling)
- 🧠 Fine-tuning of RuBERT with early stopping
- 📊 Confusion matrix & error analysis
- 🧪 Unit tests + pre-commit hooks

## Installation

```bash
git clone https://github.com/NatashaBedareva/RuBert_Classifier.git
cd RuBert_Classifier

# Install uv (once, if not installed yet)
pip install uv

# Create .venv and install all dependencies (main + dev)
uv sync

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit run -a
