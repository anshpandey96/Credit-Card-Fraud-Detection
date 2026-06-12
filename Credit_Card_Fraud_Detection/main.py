"""Command-line runner for the Credit Card Fraud Detection project."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_preprocessing import analyze_dataset, load_dataset, preprocess_data
from src.evaluation import evaluate_models, identify_best_model, metrics_table, save_results
from src.model_training import train_models
from src.visualization import (
    save_amount_by_class,
    save_class_distribution,
    save_confusion_matrix,
    save_correlation_heatmap,
    save_primary_confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BASE_DIR / "dataset" / "creditcard.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def print_dataset_summary(summary: dict) -> None:
    """Print basic dataset analysis."""
    print("\nDataset Summary")
    print("-" * 40)
    print(f"Rows: {summary['rows']}")
    print(f"Columns: {summary['columns']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Duplicate rows: {summary['duplicates']}")
    print(f"Class counts: {summary['class_counts']}")


def run_pipeline(dataset_path: Path, use_demo_if_missing: bool = True) -> None:
    """Run the complete ML workflow and save all outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Credit Card Fraud Detection")
    print("=" * 40)

    df = load_dataset(dataset_path, use_demo_if_missing=use_demo_if_missing)
    summary = analyze_dataset(df)
    print_dataset_summary(summary)

    print("\nSaving visualizations...")
    save_class_distribution(df, OUTPUT_DIR)
    save_correlation_heatmap(df, OUTPUT_DIR)
    save_amount_by_class(df, OUTPUT_DIR)

    print("\nPreprocessing data with StandardScaler and SMOTE...")
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    print(f"Balanced training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    print("\nTraining Logistic Regression and Random Forest...")
    models = train_models(X_train, y_train)

    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test)
    comparison = metrics_table(results)
    best_model_name, best_metrics = identify_best_model(results)

    print("\nModel Comparison")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest Performing Model: {best_model_name}")

    save_results(results, OUTPUT_DIR / "model_results.txt")
    for model_name, metrics in results.items():
        save_confusion_matrix(metrics["confusion_matrix"], model_name, OUTPUT_DIR)
    save_primary_confusion_matrix(best_metrics["confusion_matrix"], OUTPUT_DIR)

    print(f"\nAll outputs saved in: {OUTPUT_DIR}")
    print("Pipeline completed successfully.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to creditcard.csv",
    )
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Fail if Kaggle CSV is missing instead of using generated demo data.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    run_pipeline(args.data_path, use_demo_if_missing=not args.no_demo)


if __name__ == "__main__":
    main()
