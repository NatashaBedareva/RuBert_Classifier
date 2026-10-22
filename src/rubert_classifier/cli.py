"""CLI entrypoint."""

import json
import subprocess
import sys
from pathlib import Path

import click

from rubert_classifier.config import Config
from rubert_classifier.data import dvc_pull
from rubert_classifier.inference import Predictor
from rubert_classifier.train import train_model


@click.group()
def main():
    """RuBERT text classifier CLI."""


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
@click.option("--output-dir", default="./output")
@click.option("--best-model-dir", default="./best_model")
@click.option("--epochs", default=20, type=int)
def train(config_path, output_dir, best_model_dir, epochs):
    """Train the model."""
    cfg = Config.load(config_path) if config_path else Config()
    cfg.output_dir = output_dir
    cfg.best_model_dir = best_model_dir
    cfg.num_train_epochs = epochs

    # DVC pull happens inside get_training_datasets -> _load_sib200_dataset
    train_model(cfg)
    click.echo("Training complete.")


@main.command()
@click.option("--model-dir", required=True, type=click.Path())
@click.option("--text", default=None, help="Single text to classify.")
@click.option("--file", "file_path", default=None, type=click.Path(exists=True))
def predict(model_dir, text, file_path):
    """Run inference."""
    # Pull DVC-tracked artifacts
    dvc_pull()

    predictor = Predictor(model_dir)

    if text:
        results = predictor.predict(text)
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
    elif file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        results = predictor.predict(texts)
        for r in results:
            click.echo(f"[{r['label']}] ({r['confidence']:.3f}) {r['text']}")
    else:
        # Interactive
        click.echo("Enter text (Ctrl+C to exit):")
        try:
            while True:
                line = click.prompt(">", prompt_suffix=" ")
                if not line.strip():
                    continue
                r = predictor.predict(line)[0]
                click.echo(f"  -> {r['label']} (confidence: {r['confidence']:.3f})")
        except (KeyboardInterrupt, click.Abort):
            click.echo("\nBye.")


def ensure_data_available():
    """Ensure DVC-tracked data is pulled locally before training/inference."""
    marker = Path("data/sib200_rus")
    if marker.exists():
        return

    click.echo("[DVC] Data not found, pulling...")
    result = subprocess.run([sys.executable, "-m", "dvc", "pull"], capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"[DVC] Pull failed: {result.stderr}", err=True)
        raise click.Abort()
    click.echo("[DVC] Data ready.")


if __name__ == "__main__":
    main()
