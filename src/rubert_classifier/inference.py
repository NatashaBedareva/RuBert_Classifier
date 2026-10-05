"""Production inference module."""

import warnings
from typing import List, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rubert_classifier.model import get_device

# Suppress false-positive warnings from transformers v5.x
warnings.filterwarnings(
    "ignore",
    message=".*incorrect regex pattern.*",
    category=UserWarning,
)


class Predictor:
    """Ready-to-use inference class."""

    def __init__(self, model_dir: str, device: str = None):
        self.model_dir = model_dir
        self.device = torch.device(device) if device else get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # id2label is always available in the model config
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, texts: Union[str, List[str]], max_length: int = 256) -> List[dict]:
        """Predict categories for one or more texts."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        results = []
        for i, p in enumerate(preds):
            label = self.id2label.get(int(p), str(p))
            results.append(
                {
                    "text": texts[i],
                    "label": label,
                    "confidence": float(probs[i][p]),
                    "probabilities": {
                        self.id2label.get(j, str(j)): float(probs[i][j])
                        for j in range(probs.shape[1])
                    },
                }
            )

        return results
