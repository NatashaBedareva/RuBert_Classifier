"""Data loading module."""

from pathlib import Path
from typing import Tuple, cast

from datasets import DatasetDict, load_dataset, load_from_disk

from rubert_classifier.config import Config

DEFAULT_LOCAL_DATA_PATH = "data/sib200_rus"


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
    categories = sorted(set(dataset["train"]["category"]))
    cfg.list_of_categories = categories
    cfg.num_labels = len(categories)
    cfg.id2label = {i: c for i, c in enumerate(categories)}
    cfg.label2id = {c: i for i, c in enumerate(categories)}


def _print_stats(dataset: DatasetDict, cfg: Config) -> None:
    print(f"Dataset loaded: {cfg.dataset_name}/{cfg.dataset_language}")
    print(f"Train: {len(dataset['train'])}")
    print(f"Validation: {len(dataset['validation'])}")
    print(f"Test: {len(dataset['test'])}")
    print(f"Categories ({cfg.num_labels}): {cfg.list_of_categories}")


def _download_from_hf(cfg: Config, path: Path) -> DatasetDict:
    print(f"[data] Downloading {cfg.dataset_name}/{cfg.dataset_language} from HuggingFace...")
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_language)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    print(f"[data] Saved to {path}. Track it with `dvc add {path} && dvc push`.")
    return dataset


def _load_sib200_dataset(
    cfg: Config,
    local_path: str = DEFAULT_LOCAL_DATA_PATH,
) -> DatasetDict:
    """
    Load the Sib200 dataset.

    Order of resolution:
        1. If `force_download` is False and `local_path` exists -> load from disk.
        2. Otherwise -> load from HuggingFace and save to `local_path`
           (so that a subsequent `dvc add` / `dvc push` can track it).

    Args:
        cfg: project configuration.
        local_path: directory with the DVC-tracked local copy of the dataset.
        force_download: skip the local copy and re-fetch from HuggingFace.
    """
    path = Path(local_path)

    if _local_exists(path):
        print(f"[data] Loading local copy from {path}")
        dataset = load_from_disk(str(path))
    else:
        dvc_pull()
        if _local_exists(path):
            print(f"[data] Restored via DVC, loading from {path}")
            dataset = load_from_disk(str(path))
        else:
            dataset = _download_from_hf(cfg, path)

    dataset = cast(DatasetDict, load_from_disk(str(path)))

    _apply_categories(dataset, cfg)
    _print_stats(dataset, cfg)
    return dataset


# ----------- Public API --------------------------------------


def get_training_datasets(cfg: Config) -> Tuple:
    """
    Public entry point for training.
    DVC pull -> load dataset -> balance -> tokenize.
    """
    from rubert_classifier.preprocessing import prepare_datasets

    raw = _load_sib200_dataset(cfg)
    return prepare_datasets(raw, cfg)
