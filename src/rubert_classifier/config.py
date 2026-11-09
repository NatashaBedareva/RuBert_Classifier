"""Configuration module with Hydra Structured Configs."""

import json
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


@dataclass
class DataConfig:
    dataset_name: str = "Davlan/sib200"
    dataset_language: str = "rus_Cyrl"
    local_path: str = "data/sib200_rus"
    splits: Dict[str, str] = field(
        default_factory=lambda: {"train": "train", "validation": "validation", "test": "test"}
    )
    columns: Dict[str, str] = field(default_factory=lambda: {"text": "text", "label": "category"})
    dvc: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "remote": "local_storage"}
    )


@dataclass
class PreprocessingConfig:
    target_samples_per_class: int = 120
    max_length: int = 256
    tokenization: Dict[str, Any] = field(
        default_factory=lambda: {
            "truncation": True,
            "padding": True,
            "add_special_tokens": True,
            "pad_to_multiple_of": 8,
        }
    )
    balancing: Dict[str, Any] = field(
        default_factory=lambda: {"strategy": "oversampling", "seed": 42}
    )


@dataclass
class ModelConfig:
    name: str = "DeepPavlov/rubert-base-cased"
    num_labels: Optional[int] = None
    ignore_mismatched_sizes: bool = True
    device: str = "auto"


@dataclass
class TrainingConfig:
    learning_rate: float = 1.5e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 20
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    label_smoothing_factor: float = 0.1
    logging_steps: int = 20
    save_total_limit: int = 2
    dataloader_num_workers: int = 2
    fp16: Any = "auto"
    optimizer: str = "adamw_torch"
    scheduler: str = "cosine_with_restarts"
    strategy: Dict[str, Any] = field(
        default_factory=lambda: {
            "eval": "epoch",
            "save": "epoch",
            "load_best_model_at_end": True,
            "greater_is_better": True,
            "metric_for_best_model": "f1_macro",
        }
    )
    early_stopping: Dict[str, Any] = field(
        default_factory=lambda: {"patience": 4, "threshold": 0.001}
    )


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1_macro", "f1_weighted"])
    f1_average: str = "macro"
    classification_report_digits: int = 4
    top_errors: int = 10
    confusion_matrix: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "path": "./output/confusion_matrix.png",
            "figsize": [10, 8],
            "dpi": 300,
            "cmap": "Blues",
        }
    )


@dataclass
class InferenceConfig:
    max_length: int = 256
    batch_size: int = 32
    confidence_threshold: float = 0.0
    output_format: str = "json"


def _from_dict(cls, data):
    """Recursively convert dict -> dataclass for nested configs."""
    if not is_dataclass(cls):
        return data
    kwargs = {}
    try:
        type_hints = typing.get_type_hints(cls)
    except Exception:
        type_hints = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = type_hints.get(f.name, f.type)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


@dataclass
class Config:
    """Main configuration for training and inference."""

    defaults: List[Any] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    seed: int = 42
    output_dir: str = "./output"
    best_model_dir: str = "./best_model"
    logs_dir: str = "./logs"

    # Runtime-populated (never in YAML)
    list_of_categories: List[str] = field(default_factory=list)
    id2label: Dict[int, str] = field(default_factory=dict)
    label2id: Dict[str, int] = field(default_factory=dict)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _from_dict(cls, data)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return _from_dict(cls, data)

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "Config":
        """Convert a Hydra DictConfig into a plain Config instance."""
        plain = OmegaConf.to_container(cfg, resolve=True)
        plain.pop("_self_", None)
        plain.pop("defaults", None)
        return _from_dict(cls, plain)


ConfigStore.instance().store(name="base_config", node=Config)
