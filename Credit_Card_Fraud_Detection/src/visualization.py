"""Visualization functions for analysis and model evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid", palette="Set2")


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return the output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_class_distribution(df, output_dir: str | Path) -> Path:
    """Save class distribution chart."""
    output_dir = ensure_output_dir(output_dir)
    output_path = output_dir / "class_distribution.png"

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="Class", hue="Class", legend=False)
    ax.set_title("Class Distribution: Legitimate vs Fraud")
    ax.set_xlabel("Class (0 = Legitimate, 1 = Fraud)")
    ax.set_ylabel("Transactions")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def save_correlation_heatmap(df, output_dir: str | Path) -> Path:
    """Save feature correlation heatmap."""
    output_dir = ensure_output_dir(output_dir)
    output_path = output_dir / "correlation_heatmap.png"

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm",
        center=0,
        linewidths=0.1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def save_amount_by_class(df, output_dir: str | Path) -> Path:
    """Save transaction amount chart by class."""
    output_dir = ensure_output_dir(output_dir)
    output_path = output_dir / "amount_by_class.png"

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Class", y="Amount", hue="Class", showfliers=False, legend=False)
    plt.title("Transaction Amount Distribution by Class")
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def save_confusion_matrix(cm, model_name: str, output_dir: str | Path) -> Path:
    """Save model confusion matrix."""
    output_dir = ensure_output_dir(output_dir)
    safe_name = model_name.lower().replace(" ", "_")
    output_path = output_dir / f"confusion_matrix_{safe_name}.png"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def save_primary_confusion_matrix(cm, output_dir: str | Path) -> Path:
    """Save best-model confusion matrix using required filename."""
    output_dir = ensure_output_dir(output_dir)
    output_path = output_dir / "confusion_matrix.png"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
