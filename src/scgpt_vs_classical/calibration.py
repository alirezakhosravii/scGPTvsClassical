"""Calibration metrics, reliability diagrams, and bootstrap confidence intervals.

This module is intentionally small, dependency-light, and self-contained so
that it can be re-used by any downstream classifier (sklearn-compatible
``predict_proba`` output, raw NumPy arrays, or PyTorch logits after softmax).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------
def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : (n,) array of integer ground-truth class indices.
    y_proba : (n, K) array of predicted probabilities for ``K`` classes.
        Each row should sum to one (we do not re-normalise).
    n_bins : number of equally-spaced confidence bins on ``[0, 1]``.

    Returns
    -------
    float
        ECE in ``[0, 1]``. Lower is better; 0 indicates perfect calibration.

    Notes
    -----
    Implements the standard equation:

        ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|

    where ``B_m`` is the set of samples whose maximum predicted probability
    falls in bin ``m``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim != 2:
        raise ValueError("y_proba must be a 2D (n, K) array of probabilities.")

    n = y_proba.shape[0]
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Right-closed last bin so a confidence of exactly 1.0 is counted.
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------
@dataclass
class ReliabilityCurve:
    """Pre-computed reliability diagram for one classifier."""

    bin_centers: np.ndarray   # (n_bins,)
    bin_accuracy: np.ndarray  # (n_bins,) NaN where the bin is empty
    bin_count: np.ndarray     # (n_bins,) integer
    ece: float


def reliability_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityCurve:
    """Compute a reliability-diagram curve plus its ECE."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    accs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            accs[i] = accuracies[mask].mean()

    return ReliabilityCurve(
        bin_centers=centers,
        bin_accuracy=accs,
        bin_count=counts,
        ece=expected_calibration_error(y_true, y_proba, n_bins=n_bins),
    )


# ---------------------------------------------------------------------------
# Non-parametric bootstrap CI for Macro F1
# ---------------------------------------------------------------------------
def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = 42,
    labels: Sequence[int] | None = None,
) -> tuple[float, float, float]:
    """Bootstrap a (1-alpha) confidence interval for Macro F1.

    Parameters
    ----------
    y_true, y_pred : (n,) arrays of integer ground-truth and predicted labels.
    n_resamples : number of bootstrap resamples (paper uses 1000).
    alpha : significance level; 0.05 yields a 95% CI.
    random_state : seed for reproducibility.
    labels : the full set of class labels; pass this when some class might
        be absent from a particular bootstrap resample so that per-class F1
        is computed against a stable label set.

    Returns
    -------
    point, lo, hi : the point estimate and the (alpha/2, 1-alpha/2)
        empirical-percentile bootstrap CI bounds.
    """
    from sklearn.metrics import f1_score  # local import keeps module light

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    point = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    rng = np.random.default_rng(random_state)
    scores = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(
            y_true[idx], y_pred[idx],
            average="macro", labels=labels, zero_division=0,
        )

    lo = float(np.quantile(scores, alpha / 2))
    hi = float(np.quantile(scores, 1 - alpha / 2))
    return float(point), lo, hi
