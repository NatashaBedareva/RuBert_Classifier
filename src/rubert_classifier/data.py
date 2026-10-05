"""Data loading module."""
from datasets import DatasetDict, load_dataset

from rubert_classifier.config import Config


def load_sib200_dataset(cfg: Config) -> DatasetDict:
    """Load the Sib200 dataset."""
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_language)

    # Set categories
    categories = sorted(set(dataset["train"]["category"]))
    cfg.list_of_categories = categories
    cfg.num_labels = len(categories)
    cfg.id2label = {i: c for i, c in enumerate(categories)}
    cfg.label2id = {c: i for i, c in enumerate(categories)}

    print(f"Dataset loaded: {cfg.dataset_name}/{cfg.dataset_language}")
    print(f"Train: {len(dataset['train'])}")
    print(f"Validation: {len(dataset['validation'])}")
    print(f"Test: {len(dataset['test'])}")
    print(f"Categories ({cfg.num_labels}): {categories}")

    return dataset
