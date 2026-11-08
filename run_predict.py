"""Hydra entry point for inference."""

import json

import hydra
from omegaconf import DictConfig

from rubert_classifier.config import Config
from rubert_classifier.inference import Predictor


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    app_cfg = Config.from_hydra(cfg)
    predictor = Predictor(app_cfg.best_model_dir, inference_cfg=app_cfg.inference)

    text = "Это тестовый текст про спорт"
    results = predictor.predict(text)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()