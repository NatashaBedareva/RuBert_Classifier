"""Model creation and loading."""

import torch
from transformers import AutoModelForSequenceClassification

from rubert_classifier.config import Config


def create_model(cfg: Config):
    """Create a fresh model from a pretrained checkpoint."""
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.model.num_labels,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
        ignore_mismatched_sizes=cfg.model.ignore_mismatched_sizes,
    )
    print(f"Model created: {cfg.model.name} with {cfg.model.num_labels} labels")
    return model


def load_model(model_dir: str):
    """Load a fine-tuned model from disk."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model


def get_device(cfg: Config | None = None) -> torch.device:
    """Get the best available device according to config (auto/cuda/cpu)."""
    requested = "auto" if cfg is None else cfg.model.device
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)