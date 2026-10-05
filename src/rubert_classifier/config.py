"""Configuration module."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import json
import yaml


@dataclass
class Config:
    """Main configuration for training and inference."""

    # Data
    dataset_name: str = "Davlan/sib200"
    dataset_language: str = "rus_Cyrl"
    target_samples_per_class: int = 120
    max_length: int = 256
    seed: int = 42

    # Model
    model_name: str = "DeepPavlov/rubert-base-cased"
    num_labels: Optional[int] = None

    # Training
    output_dir: str = "./output"
    learning_rate: float = 1.5e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 20
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    label_smoothing_factor: float = 0.1
    early_stopping_patience: int = 4
    early_stopping_threshold: float = 0.001
    metric_for_best_model: str = "f1_macro"
    logging_steps: int = 20
    save_total_limit: int = 2

    # Paths
    best_model_dir: str = "./best_model"
    logs_dir: str = "./logs"

    # Categories (filled at runtime)
    list_of_categories: List[str] = field(default_factory=list)
    id2label: dict = field(default_factory=dict)
    label2id: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save config to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load config from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)