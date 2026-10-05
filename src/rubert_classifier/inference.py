"""Production inference module."""

from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rubert_classifier.config import Config
from rubert_classifier.model import get_device


class Predictor:
    """Ready-to-use inference class."""

    def __init__(self, model_dir: str, device: str = None):
        self.model_dir = model_dir
        self.device = torch.device(device) if device else get_device()

        # Load config if present
        cfg_path = Path(model_dir) / "config.json"
        self.config = Config.load(str(cfg_path)) if cfg_path.exists() else Config()

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.config.id2label if self.config.id2label else self.model.config.id2label

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
