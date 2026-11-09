# RuBERT Classifier

Text classification on the [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) dataset (Russian) using `DeepPavlov/rubert-base-cased`.

## Features

- 📦 Installable Python package
- ⚡ Fast dependency management with [uv](https://docs.astral.sh/uv/)
- 🛠️ `Makefile` shortcuts for every workflow (install / train / predict / test / sweep)
- 🚀 CLI for training and inference
- 🗂️ Data & model versioning with [DVC](https://dvc.org/)
- ⚖️ Automatic class balancing (oversampling)
- 🧠 Fine-tuning of RuBERT with early stopping
- 📊 Confusion matrix & error analysis
- 🧪 Unit tests + pre-commit hooks

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

### Installing `uv`

`uv` is the package manager used by every target in the `Makefile`. Install it first — otherwise `make install` will fail with `CreateProcess(NULL, uv sync, ...) failed / make (e=2)`.

```bash
pip install uv
```

### Cloning & installing dependencies

```bash
git clone https://github.com/NatashaBedareva/RuBert_Classifier.git
cd RuBert_Classifier

# Runtime + dev dependencies + pre-commit hooks
make install-dev
```

`make install` installs only runtime dependencies (no dev tools, no hooks).

## Makefile commands

All common workflows are wrapped in `make` targets:

| Target             | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| `make install`     | Install runtime dependencies (`uv sync`)                    |
| `make install-dev` | Install runtime + dev deps and pre-commit hooks             |
| `make test`        | Run `pytest tests/ -v`                                      |
| `make lint`        | `flake8` + `mypy`                                           |
| `make format`      | `black` + `isort`                                           |
| `make train`       | Train with default Hydra config                             |
| `make train-quick` | Fast smoke-test preset (`training=quick`, 10 samples/class) |
| `make train-full`  | Full training preset                                        |
| `make predict`     | Inference via Hydra entry point (`run_predict.py`)          |
| `make predict-cli` | Inference via `rubert-clf` CLI (interactive mode)           |
| `make sweep`       | Hydra multirun learning-rate sweep                          |
| `make clean`       | Remove caches and build artifacts                           |
| `make help`        | List available targets                                      |

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

| Artifact                           | Path               | Tracked by            |
| ---------------------------------- | ------------------ | --------------------- |
| Dataset (Sib200, Russian)          | `data/sib200_rus/` | `data/sib200_rus.dvc` |
| Best fine-tuned model (in process) | `best_model/`      | `best_model.dvc`      |

### Bootstrapping the DVC cache for the first time

On the very first run the local copy of the dataset does not exist yet. `data.py` will download it from HuggingFace and save it to `data/sib200_rus/`. Register that copy with DVC **once**:

```bash
dvc add data/sib200_rus
dvc push
git add data/sib200_rus.dvc data/.gitignore
git commit -m "data: track sib200 local copy with DVC"
```

After this, every fresh clone gets the exact same data via `dvc pull` (or automatically via the CLI / `make` targets).

### How it plugs into the code

- `data.py` is the **only** module allowed to touch DVC and datasets. Its public API is:
  - `dvc_pull()` — safe wrapper around `dvc.repo.Repo().pull()`
  - `get_training_datasets(cfg)` — DVC pull → load dataset → balance → tokenize
- `train.py` calls `get_training_datasets(cfg)`.
- `cli.py` calls `dvc_pull()` before inference so that `--model-dir` can be restored from DVC if missing.

## Training

```bash
# Default preset
make train

# Quick smoke-test preset (few samples per class)
make train-quick

# Full preset
make train-full

# Fine-grained Hydra overrides
uv run python run_train.py training.num_train_epochs=1
uv run python run_train.py --cfg job      # inspect the composed config
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

### Learning-rate sweep

```bash
# Hydra multirun over three LRs; each run writes to its own best_model dir
make sweep

# Equivalent, with explicit args
uv run python run_train.py -m training=quick preprocessing.target_samples_per_class=10 \
  training.learning_rate=1e-5,3e-5,5e-5 \
  best_model_dir=./best_models/lr_${training.learning_rate}
```

## Inference

### Via Makefile

```bash
make predict       # Hydra entry point (run_predict.py)
make predict-cli   # rubert-clf predict --model-dir ./best_model (interactive)
```

### Via CLI directly

```bash
# Single text
uv run rubert-clf predict --model-dir ./best_model --text "Ваш текст здесь"

# Batch from file (one text per line)
uv run rubert-clf predict --model-dir ./best_model --file input.txt

# Interactive mode
uv run rubert-clf predict --model-dir ./best_model
```

### Python API

```python
from rubert_classifier import Predictor, dvc_pull

dvc_pull()  # optional: only needed if best_model was fetched via DVC

predictor = Predictor("./best_model")
results = predictor.predict("Это тестовый текст")
print(results[0]["label"], results[0]["confidence"])
```

## Testing & code quality

```bash
make test     # pytest tests/ -v
make lint     # flake8 + mypy
make format   # black + isort
make clean    # drop caches and build artifacts
```

## Hydra entry points

The `Makefile` targets above are thin wrappers around the Hydra scripts; the raw commands remain available for experimentation:

```bash
# Training
uv run python run_train.py
uv run python run_train.py training=quick
uv run python run_train.py training=full
uv run python run_train.py training.num_train_epochs=1
uv run python run_train.py --cfg job          # inspect the composed config
uv run python run_train.py -m training=quick,full

# Inference
uv run python run_predict.py
uv run python run_predict.py inference.max_length=128
```

## Project structure

```
Makefile                # install / train / predict / sweep / clean shortcuts

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

configs/
├── config.yaml       # root: defaults + global fields
├── data/sib200.yaml  # dataset, splits, columns, DVC
├── preprocessing/default.yaml # balancing, tokenization
├── model/rubert.yaml # model name, device
├── training/
│ ├── default.yaml    # default training params
│ ├── quick.yaml      # fast smoke-test preset
│ └── full.yaml       # full training preset
├── evaluation/default.yaml # metrics, confusion matrix
├── inference/default.yaml # max_length, confidence threshold
└── serving/local.yaml # (reserved for serving)

run_train.py # Hydra entry point for training
run_predict.py # Hydra entry point for inference
```

## License

MIT
