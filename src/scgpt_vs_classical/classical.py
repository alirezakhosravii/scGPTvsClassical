"""Classical machine-learning baselines used in the benchmark.

Each baseline is wrapped to a uniform interface that returns:

    {
        "model"  : the trained sklearn-compatible estimator,
        "y_pred" : (n_test,) ground-truth-aligned predicted labels,
        "y_proba": (n_test, n_classes) predicted probabilities,
    }

so the rest of the pipeline (calibration metrics, plotting, reporting)
does not need to know which family produced the predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _pack(model, X_test) -> dict[str, Any]:
    return {
        "model": model,
        "y_pred": model.predict(X_test),
        "y_proba": model.predict_proba(X_test),
    }


def train_logreg(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """L2-regularised multinomial Logistic Regression."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        n_jobs=n_jobs,
        multi_class="auto",
    )
    model.fit(X_train, y_train)
    return _pack(model, X_test)


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Multi-class XGBoost with multinomial log-loss."""
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return _pack(model, X_test)


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_estimators: int = 200,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Random Forest with sklearn defaults plus parallel training."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return _pack(model, X_test)
