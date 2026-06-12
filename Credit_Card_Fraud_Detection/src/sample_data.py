"""Demo data generator used when the Kaggle CSV is not available locally."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_dataset(n_samples: int = 2500, fraud_ratio: float = 0.035) -> pd.DataFrame:
    """Create a Kaggle-like synthetic dataset for local demos and smoke tests."""
    rng = np.random.default_rng(42)
    n_fraud = max(20, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    legitimate = rng.normal(0, 1, size=(n_legit, len(columns)))
    fraud = rng.normal(0, 1, size=(n_fraud, len(columns)))

    fraud[:, 1] += 2.4
    fraud[:, 3] -= 2.0
    fraud[:, 8] += 1.8
    fraud[:, 14] -= 2.5
    fraud[:, 29] = np.abs(rng.normal(180, 80, n_fraud))
    legitimate[:, 29] = np.abs(rng.normal(65, 45, n_legit))

    X = np.vstack([legitimate, fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)

    df = pd.DataFrame(X, columns=columns)
    df["Time"] = np.abs(df["Time"] * 10000).astype(int)
    df["Amount"] = df["Amount"].clip(lower=0)
    df["Class"] = y
    return df.sample(frac=1, random_state=42).reset_index(drop=True)
