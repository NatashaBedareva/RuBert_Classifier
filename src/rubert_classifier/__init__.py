"""RuBERT text classifier package."""

from rubert_classifier.config import Config
from rubert_classifier.data import load_sib200_dataset
from rubert_classifier.inference import Predictor
from rubert_classifier.model import create_model, load_model
from rubert_classifier.train import train_model

__version__ = "0.1.0"

__all__ = [
    "Config",
    "create_model",
    "load_model",
    "load_sib200_dataset",
    "train_model",
    "Predictor",
]
