"""MLflow integration: callbacks, plot generation, config logging."""

import subprocess
from dataclasses import asdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
from transformers import TrainerCallback

from rubert_classifier.config import Config

matplotlib.use("Agg")  # безголовый режим


def get_git_commit() -> str:
    """Возвращает hash текущего коммита или 'unknown'."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Рекурсивно делает плоский dict из вложенного (MLflow не любит вложенность)."""
    flat = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, new_key))
        elif isinstance(value, list):
            flat[new_key] = str(value)
        else:
            flat[new_key] = value
    return flat


def flatten_config(cfg: Config) -> dict:
    """Превращает Config в плоский dict для mlflow.log_params."""
    return _flatten_dict(asdict(cfg))


def setup_mlflow(cfg: Config) -> str:
    """Настраивает MLflow и возвращает имя рана."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    return (
        cfg.mlflow.run_name
        or f"train_{cfg.training.num_train_epochs}ep_lr{cfg.training.learning_rate}"
    )


class MLflowPlotCallback(TrainerCallback):
    """
    Логирует метрики в MLflow и по завершении обучения сохраняет графики в plots/.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history = {"step": [], "loss": [], "eval_accuracy": [], "eval_f1_macro": []}

        # Уникальная папка для каждого рана (по run_id), чтобы избежать перезаписи
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "unknown"
        self.plots_dir = Path(cfg.plots_dir) / run_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        mlflow.log_metrics(logs, step=state.global_step)

        step = state.global_step
        if "loss" in logs:
            self.history["step"].append(step)
            self.history["loss"].append(logs["loss"])
        if "eval_accuracy" in logs:
            self.history["eval_accuracy"].append(logs["eval_accuracy"])
            self.history["eval_f1_macro"].append(logs.get("eval_f1_macro"))

    def on_train_end(self, args, state, control, **kwargs):
        """Сохраняет графики в plots/<run_id>/ и логирует их в MLflow."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # 1. Loss
        if self.history["loss"]:
            plt.figure(figsize=(10, 5))
            plt.plot(
                self.history["step"], self.history["loss"], label="train loss", color="tab:blue"
            )
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            loss_path = self.plots_dir / "loss.png"
            plt.tight_layout()
            plt.savefig(loss_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(loss_path), artifact_path="plots")

        # 2. Accuracy
        if self.history["eval_accuracy"]:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history["eval_accuracy"], label="eval accuracy", color="tab:green")
            plt.xlabel("Eval step")
            plt.ylabel("Accuracy")
            plt.title("Validation Accuracy")
            plt.grid(True, alpha=0.3)
            plt.legend()
            acc_path = self.plots_dir / "accuracy.png"
            plt.tight_layout()
            plt.savefig(acc_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(acc_path), artifact_path="plots")

        # 3. F1 macro
        f1_vals = [v for v in self.history["eval_f1_macro"] if v is not None]
        if f1_vals:
            plt.figure(figsize=(10, 5))
            plt.plot(f1_vals, label="eval f1_macro", color="tab:red")
            plt.xlabel("Eval step")
            plt.ylabel("F1 macro")
            plt.title("Validation F1 (macro)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            f1_path = self.plots_dir / "f1_macro.png"
            plt.tight_layout()
            plt.savefig(f1_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(f1_path), artifact_path="plots")
