"""Evaluation and model comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate one trained classifier."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        ),
        "predictions": y_pred,
    }


def evaluate_models(models: Dict[str, object], X_test, y_test) -> Dict[str, dict]:
    """Evaluate every trained model."""
    return {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}


def metrics_table(results: Dict[str, dict]) -> pd.DataFrame:
    """Return a clean model comparison table."""
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
            }
        )
    return pd.DataFrame(rows).sort_values("F1-Score", ascending=False)


def identify_best_model(results: Dict[str, dict]) -> Tuple[str, dict]:
    """Select the best model using F1-score."""
    best_name = max(results, key=lambda name: results[name]["f1_score"])
    return best_name, results[best_name]


def save_results(results: Dict[str, dict], output_path: str | Path) -> None:
    """Write metrics and classification reports to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = metrics_table(results)
    best_model, _ = identify_best_model(results)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Credit Card Fraud Detection - Model Results\n")
        file.write("=" * 52 + "\n\n")
        file.write("Model Comparison\n")
        file.write(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        file.write("\n\n")
        file.write(f"Best Performing Model: {best_model}\n")
        file.write("Selection Metric: Highest F1-Score\n\n")

        for model_name, metrics in results.items():
            file.write(f"{model_name}\n")
            file.write("-" * len(model_name) + "\n")
            file.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
            file.write(f"Precision: {metrics['precision']:.4f}\n")
            file.write(f"Recall   : {metrics['recall']:.4f}\n")
            file.write(f"F1-Score : {metrics['f1_score']:.4f}\n")
            file.write("Confusion Matrix:\n")
            file.write(f"{metrics['confusion_matrix']}\n\n")
            file.write("Classification Report:\n")
            file.write(metrics["classification_report"])
            file.write("\n\n")
