"""RuBERT text classifier package."""

from rubert_classifier.config import Config
from rubert_classifier.data import dvc_pull, get_training_datasets
from rubert_classifier.inference import Predictor
from rubert_classifier.model import create_model, load_model
from rubert_classifier.train import train_model

__version__ = "0.1.1"

__all__ = [
    "Config",
    "create_model",
    "load_model",
    "dvc_pull",
    "get_training_datasets",
    "train_model",
    "Predictor",
]
