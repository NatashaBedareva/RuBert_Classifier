# RuBERT Classifier

Text classification on the [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) dataset (Russian) using `DeepPavlov/rubert-base-cased`.

## Features

- 📦 Installable Python package
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

python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# or: .venv\Scripts\activate # Windows

pip install -e ".[dev]"
pre-commit install
pre-commit run -a
```

## Data management with DVC

The dataset is tracked with DVC so that every experiment uses a reproducible version of the data. Pulling is **built into the CLI** (`train` and `predict`) and into `data.py`, so you normally don't need to run `dvc pull` manually — it happens automatically on the first call.

### First-time setup (once per clone)

```bash
# Configure a default remote. Examples:
#   local storage
dvc remote add -d local_storage /tmp/dvc-storage-rubert
#   or S3 (make sure `dvc[s3]` is installed and credentials are set)
# dvc remote add -d s3remote s3://<your-bucket>/rubert-data

# Pull data + model artifacts tracked by DVC
dvc pull
```

If `dvc pull` fails because the remote is empty (fresh project), the loader will fall back to downloading the dataset from HuggingFace automatically — see below.

### What is tracked

| Artifact | Path | Tracked by |
|---|---|---|
| Dataset (Sib200, Russian) | `data/sib200_rus/` | `data/sib200_rus.dvc` |
| Best fine-tuned model (in process) | `best_model/` | `best_model.dvc` |

### Bootstrapping the DVC cache for the first time

On the very first run the local copy of the dataset does not exist yet. `data.py` will download it from HuggingFace and save it to `data/sib200_rus/`. Register that copy with DVC **once**:

```bash
dvc add data/sib200_rus
dvc push
git add data/sib200_rus.dvc data/.gitignore
git commit -m "data: track sib200 local copy with DVC"
```

After this, every fresh clone gets the exact same data via `dvc pull` (or automatically via the CLI).

### How it plugs into the code

- `data.py` is the **only** module allowed to touch DVC and datasets. Its public API is:
  - `dvc_pull()` — safe wrapper around `dvc.repo.Repo().pull()`
  - `get_training_datasets(cfg)` — DVC pull → load dataset → balance → tokenize
- `train.py` calls `get_training_datasets(cfg)`.
- `cli.py` calls `dvc_pull()` before inference so that `--model-dir` can be restored from DVC if missing.

## Training

```bash
# Using CLI
rubert-clf train --output-dir ./output --best-model-dir ./best_model --epochs 20
```

Or from Python:

```python
from rubert_classifier import Config, get_training_datasets, train_model

cfg = Config()
trainer = train_model(cfg)
```

After training, register the best model with DVC:

```bash
dvc add best_model
dvc push
git add best_model.dvc best_model/.gitignore
git commit -m "model: track best_model with DVC"
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
from rubert_classifier import Predictor, dvc_pull

dvc_pull()  # optional: only needed if best_model was fetched via DVC

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
├── data.py           # DVC pull + dataset loading (the ONLY place touching DVC/data)
├── preprocessing.py  # Balancing & tokenization
├── model.py          # Model creation/loading
├── train.py          # Training pipeline
├── inference.py      # Predictor class
├── evaluation.py     # Metrics & error analysis
├── cli.py            # Command-line interface
└── utils.py          # Helpers

data/
├── sib200_rus/       # DVC-tracked dataset (locally)
└── sib200_rus.dvc    # DVC pointer (git)

.dvc/
└── config            # DVC remote configuration (git)
```


## License

MIT
