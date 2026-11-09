"""Data loading module."""

from pathlib import Path
from typing import Tuple, cast

from datasets import DatasetDict, load_dataset, load_from_disk

from rubert_classifier.config import Config


# --------------- DVC --------------------------------
def dvc_pull() -> bool:
    """Pull DVC-tracked artifacts (data + models). Returns True on success."""
    try:
        from dvc.repo import Repo
    except ImportError:
        print("[dvc] package not installed — skipping pull.")
        return False

    print("[dvc] Pulling tracked artifacts...")
    try:
        pulled = Repo(".").pull()
        if not pulled:
            print("[dvc] Nothing to pull (already up to date).")
        return True
    except Exception as exc:
        print(f"[dvc] pull failed: {exc}")
        return False


# ------------- Internals -----------------------------------------
def _local_exists(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _apply_categories(dataset: DatasetDict, cfg: Config) -> None:
    train_split = cfg.data.splits["train"]
    label_col = cfg.data.columns["label"]
    categories = sorted(set(dataset[train_split][label_col]))
    cfg.list_of_categories = categories
    cfg.model.num_labels = len(categories)
    cfg.id2label = {i: c for i, c in enumerate(categories)}
    cfg.label2id = {c: i for i, c in enumerate(categories)}


def _print_stats(dataset: DatasetDict, cfg: Config) -> None:
    print(f"Dataset loaded: {cfg.data.dataset_name}/{cfg.data.dataset_language}")
    for split_name, split_key in cfg.data.splits.items():
        print(f"{split_name.capitalize()}: {len(dataset[split_key])}")
    print(f"Categories ({cfg.model.num_labels}): {cfg.list_of_categories}")


def _download_from_hf(cfg: Config, path: Path) -> DatasetDict:
    print(
        f"[data] Downloading {cfg.data.dataset_name}/{cfg.data.dataset_language} "
        "from HuggingFace..."
    )
    dataset = load_dataset(cfg.data.dataset_name, cfg.data.dataset_language)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    print(f"[data] Saved to {path}. Track it with `dvc add {path} && dvc push`.")
    return dataset


def _load_sib200_dataset(cfg: Config) -> DatasetDict:
    """
    Load the Sib200 dataset.

    Order of resolution:
        1. If local copy exists -> load from disk.
        2. Else -> dvc pull (if enabled); if restored -> load from disk.
        3. Else -> download from HuggingFace and save locally.
    """
    path = Path(cfg.data.local_path)

    if _local_exists(path):
        print(f"[data] Loading local copy from {path}")
        return cast(DatasetDict, load_from_disk(str(path)))

    if cfg.data.dvc.get("enabled", True):
        dvc_pull()
        if _local_exists(path):
            print(f"[data] Restored via DVC, loading from {path}")
            return cast(DatasetDict, load_from_disk(str(path)))

    dataset = _download_from_hf(cfg, path)
    return cast(DatasetDict, dataset)


# ----------- Public API --------------------------------------


def get_training_datasets(cfg: Config) -> Tuple:
    """
    Public entry point for training.
    DVC pull -> load dataset -> balance -> tokenize.
    """
    from rubert_classifier.preprocessing import prepare_datasets

    raw = _load_sib200_dataset(cfg)
    _apply_categories(raw, cfg)
    _print_stats(raw, cfg)
    return prepare_datasets(raw, cfg)
