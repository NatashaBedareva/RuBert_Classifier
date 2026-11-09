"""Production inference module."""

import warnings
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rubert_classifier.config import InferenceConfig
from rubert_classifier.model import get_device

warnings.filterwarnings(
    "ignore",
    message=".*incorrect regex pattern.*",
    category=UserWarning,
)


class Predictor:
    """Ready-to-use inference class."""

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        inference_cfg: Optional[InferenceConfig] = None,
    ):
        self.model_dir = model_dir
        self.inference_cfg = inference_cfg or InferenceConfig()
        self.device = torch.device(device) if device else get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, texts: Union[str, List[str]], max_length: Optional[int] = None) -> List[dict]:
        """Predict categories for one or more texts."""
        max_length = max_length or self.inference_cfg.max_length
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
            confidence = float(probs[i][p])
            if confidence < self.inference_cfg.confidence_threshold:
                label = "unknown"
            results.append(
                {
                    "text": texts[i],
                    "label": label,
                    "confidence": confidence,
                    "probabilities": {
                        self.id2label.get(j, str(j)): float(probs[i][j])
                        for j in range(probs.shape[1])
                    },
                }
            )

        return results
