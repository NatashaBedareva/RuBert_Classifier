# RuBERT Sib200 Text Classifier

Text classification on the [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) dataset (Russian) using `DeepPavlov/rubert-base-cased`.

## Features

- 📦 Installable Python package
- 🚀 CLI for training and inference
- ⚖️ Automatic class balancing (oversampling)
- 🧠 Fine-tuning of RuBERT with early stopping
- 📊 Confusion matrix & error analysis
- 🧪 Unit tests + pre-commit hooks

## Installation

```bash
git clone https://github.com/<your-username>/rubert-sib200-classifier.git
cd rubert-sib200-classifier

python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# or: .venv\Scripts\activate # Windows

pip install -e ".[dev]"
pre-commit install
pre-commit run -a
```

## Training

```bash
# Using CLI
rubert-clf train --output-dir ./output --best-model-dir ./best_model --epochs 20
```

Or from Python:

```python
from rubert_classifier import Config, load_sib200_dataset, train_model

cfg = Config()
dataset = load_sib200_dataset(cfg)
trainer = train_model(cfg, dataset)
```

## Inference

### CLI

```bash
# Single text
rubert-clf predict --model-dir ./best_model --text "Ваш текст здесь"

# Batch from file (one text per line)
rubert-clf predict --model-dir ./best_model --file input.txt

# Interactive mode
rubert-clf predict --model-dir ./best_model
```

### Python API

```python
from rubert_classifier import Predictor

predictor = Predictor("./best_model")
results = predictor.predict("Это тестовый текст")
print(results[0]["label"], results[0]["confidence"])
```

## Testing

```bash
pytest tests/ -v
```

## Project structure

```
src/rubert_classifier/
├── config.py         # Configuration dataclass
├── data.py           # Dataset loading
├── preprocessing.py  # Balancing & tokenization
├── model.py          # Model creation/loading
├── train.py          # Training pipeline
├── inference.py      # Predictor class
├── evaluation.py     # Metrics & error analysis
├── cli.py            # Command-line interface
└── utils.py          # Helpers
```

## License

MIT