"""Data loading, cleaning, scaling, balancing, and splitting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.sample_data import generate_demo_dataset


TARGET_COLUMN = "Class"
DEFAULT_RANDOM_STATE = 42


def load_dataset(csv_path: str | Path, use_demo_if_missing: bool = True) -> pd.DataFrame:
    """Load the Kaggle dataset, or return demo data if the CSV is unavailable."""
    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("The dataset file is empty.")
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Expected target column '{TARGET_COLUMN}' was not found.")
        return df

    if use_demo_if_missing:
        print(
            "Dataset file not found. Using generated demo data so the project can run. "
            "For final results, place Kaggle creditcard.csv inside the dataset folder."
        )
        return generate_demo_dataset()

    raise FileNotFoundError(
        f"Dataset not found at {path}. Download creditcard.csv from "
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it inside "
        "the dataset folder."
    )


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Return high-level dataset facts for reporting and console output."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "class_counts": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and remove duplicate rows."""
    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    cleaned[numeric_columns] = cleaned[numeric_columns].fillna(
        cleaned[numeric_columns].median()
    )

    non_numeric_columns = cleaned.columns.difference(numeric_columns)
    for column in non_numeric_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].mode().iloc[0])

    return cleaned.drop_duplicates().reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature columns from the binary target."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train-test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler fitted only on training data."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance the training set with SMOTE."""
    minority_count = int(y_train.value_counts().min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least two minority-class samples.")

    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Run cleaning, splitting, scaling, and SMOTE balancing."""
    cleaned = clean_dataset(df)
    X, y = split_features_target(cleaned)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_balanced, y_train_balanced = apply_smote(
        X_train_scaled, y_train, random_state=random_state
    )
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler
