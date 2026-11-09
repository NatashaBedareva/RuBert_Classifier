"""Evaluation and error analysis."""

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from rubert_classifier.config import EvaluationConfig


def evaluate_predictions(
    true_labels: List[int],
    pred_labels: List[int],
    list_of_categories: List[str],
    cfg: Optional[EvaluationConfig] = None,
) -> dict:
    """Compute detailed metrics."""
    cfg = cfg or EvaluationConfig()

    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted")
    accuracy = float(np.mean(np.asarray(true_labels) == np.asarray(pred_labels)))

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=list_of_categories,
        digits=cfg.classification_report_digits,
    )

    print(f"\nMacro F1:    {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print("\nDetailed report:")
    print(report)

    return {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "accuracy": accuracy,
        "report": report,
    }


def analyze_errors(
    true_labels: List[int],
    pred_labels: List[int],
    test_texts: List[str],
    list_of_categories: List[str],
    id2label: dict,
    cfg: Optional[EvaluationConfig] = None,
):
    """Detailed error analysis + confusion matrix."""
    cfg = cfg or EvaluationConfig()
    cm_cfg = cfg.confusion_matrix

    cm = confusion_matrix(true_labels, pred_labels)

    errors = []
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        if true != pred:
            errors.append(
                {
                    "text": test_texts[i],
                    "true": id2label[int(true)],
                    "pred": id2label[int(pred)],
                }
            )

    error_types = {}
    for e in errors:
        key = f"{e['true']} -> {e['pred']}"
        error_types[key] = error_types.get(key, 0) + 1

    print("\nMost common error types:")
    for et, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {et}: {count}")

    print(f"\nSample errors ({min(cfg.top_errors, len(errors))}):")
    for i, e in enumerate(errors[: cfg.top_errors]):
        preview = e["text"][:200] + "..." if len(e["text"]) > 200 else e["text"]
        print(f"\n{i+1}. TRUE: {e['true']}, PRED: {e['pred']}")
        print(f"   {preview}")

    if cm_cfg.get("enabled", True):
        try:
            plt.figure(figsize=tuple(cm_cfg["figsize"]))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap=cm_cfg["cmap"],
                xticklabels=list_of_categories,
                yticklabels=list_of_categories,
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True")
            plt.xlabel("Predicted")
            plt.tight_layout()
            save_path = cm_cfg["path"]
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=cm_cfg["dpi"], bbox_inches="tight")
            plt.close()
            print(f"\nConfusion matrix saved to {save_path}")
        except Exception as exc:  # pragma: no cover
            print(f"Could not save confusion matrix: {exc}")

    return errors, cm
