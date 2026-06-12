"""Model training utilities."""

from __future__ import annotations

import os
from typing import Dict

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_models(random_state: int = 42) -> Dict[str, object]:
    """Create required classification models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
        ),
    }


def train_models(X_train, y_train, random_state: int = 42) -> Dict[str, object]:
    """Train all configured models."""
    trained_models = {}
    for model_name, model in build_models(random_state).items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
    return trained_models
