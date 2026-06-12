"""Streamlit dashboard for Credit Card Fraud Detection."""

from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib  # type: ignore[import]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import]
import importlib.util

_seaborn_spec = importlib.util.find_spec("seaborn")
if _seaborn_spec is not None:
    sns = importlib.import_module("seaborn")
    _HAS_SEABORN = True
else:
    sns = None
    _HAS_SEABORN = False
import numpy as np
import pandas as pd
import streamlit as st  # type: ignore[import]

from src.data_preprocessing import analyze_dataset, preprocess_data
from src.evaluation import evaluate_models, identify_best_model, metrics_table
from src.model_training import train_models
from src.sample_data import generate_demo_dataset


DATASET_PATH = BASE_DIR / "dataset" / "creditcard.csv"


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="CC",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_app_data(uploaded_file) -> pd.DataFrame:
    """Load uploaded, local, or generated demo data."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return generate_demo_dataset()


@st.cache_data(show_spinner=True)
def train_and_evaluate(df: pd.DataFrame):
    """Preprocess data, train models, and evaluate them."""
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    return results


def plot_class_distribution(df: pd.DataFrame):
    """Create class distribution figure for Streamlit."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if _HAS_SEABORN:
        sns.countplot(data=df, x="Class", hue="Class", legend=False, ax=ax)
    else:
        counts = df["Class"].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=["C0", "C1"])
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class (0 = Legitimate, 1 = Fraud)")
    ax.set_ylabel("Transactions")
    return fig


def plot_confusion_matrix(cm, title: str):
    """Create confusion matrix figure for Streamlit."""
    fig, ax = plt.subplots(figsize=(5, 4))
    if _HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax,
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legitimate", "Fraud"])
        ax.set_yticklabels(["Legitimate", "Fraud"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


st.title("Credit Card Fraud Detection")
st.caption("Logistic Regression vs Random Forest with StandardScaler and SMOTE")

uploaded_file = st.sidebar.file_uploader("Upload Kaggle creditcard.csv", type=["csv"])
st.sidebar.info(
    "If no CSV is uploaded and dataset/creditcard.csv is missing, the app uses demo data "
    "so you can still run and test the project."
)

df = load_app_data(uploaded_file)
summary = analyze_dataset(df)

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{summary['rows']:,}")
metric_cols[1].metric("Columns", summary["columns"])
metric_cols[2].metric("Missing Values", summary["missing_values"])
metric_cols[3].metric("Duplicate Rows", summary["duplicates"])

st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Class Distribution")
    st.pyplot(plot_class_distribution(df), use_container_width=True)

with right:
    st.subheader("Transaction Amount by Class")
    fig, ax = plt.subplots(figsize=(7, 4))
    if _HAS_SEABORN:
        sns.boxplot(data=df, x="Class", y="Amount", hue="Class", showfliers=False, legend=False, ax=ax)
    else:
        # Fallback to matplotlib boxplot when seaborn is not available
        grouped = [df.loc[df["Class"] == cls, "Amount"].dropna() for cls in sorted(df["Class"].unique())]
        ax.boxplot(grouped, labels=[str(c) for c in sorted(df["Class"].unique())], showfliers=False)
    ax.set_xlabel("Class (0 = Legitimate, 1 = Fraud)")
    ax.set_ylabel("Amount")
    st.pyplot(fig, use_container_width=True)

if st.button("Train and Compare Models", type="primary"):
    with st.spinner("Training models and calculating metrics..."):
        results = train_and_evaluate(df)
        comparison = metrics_table(results)
        best_model_name, best_metrics = identify_best_model(results)

    st.subheader("Model Comparison")
    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
            }
        ),
        use_container_width=True,
    )
    st.success(f"Best Performing Model: {best_model_name}")

    cm_cols = st.columns(2)
    for index, (model_name, metrics) in enumerate(results.items()):
        with cm_cols[index % 2]:
            st.pyplot(
                plot_confusion_matrix(
                    metrics["confusion_matrix"], f"Confusion Matrix - {model_name}"
                ),
                use_container_width=True,
            )
            with st.expander(f"Classification Report - {model_name}"):
                st.text(metrics["classification_report"])

    st.subheader("Best Model Confusion Matrix")
    st.pyplot(
        plot_confusion_matrix(best_metrics["confusion_matrix"], best_model_name),
        use_container_width=True,
    )
