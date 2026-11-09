"""Hydra entry point for training."""

import hydra
from omegaconf import DictConfig

from rubert_classifier.config import Config
from rubert_classifier.train import train_model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    app_cfg = Config.from_hydra(cfg)
    train_model(app_cfg)


if __name__ == "__main__":
    main()
