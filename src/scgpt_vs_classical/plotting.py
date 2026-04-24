"""Reliability diagrams and confusion matrices used in the paper figures."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .calibration import reliability_curve


def plot_reliability_grid(
    panels: Iterable[tuple[str, np.ndarray, np.ndarray]],
    n_bins: int = 10,
    n_cols: int = 3,
    suptitle: str | None = None,
):
    """Plot a grid of reliability diagrams.

    Parameters
    ----------
    panels : iterable of ``(title, y_true, y_proba)`` tuples (paper uses 6 panels).
    n_bins : number of equally-spaced confidence bins.
    n_cols : grid columns; rows are derived automatically.
    """
    import matplotlib.pyplot as plt

    panels = list(panels)
    n = len(panels)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, y_true, y_proba) in zip(axes, panels):
        rc = reliability_curve(y_true, y_proba, n_bins=n_bins)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.bar(rc.bin_centers, rc.bin_accuracy, width=1.0 / n_bins, alpha=0.65,
               edgecolor="black", linewidth=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"{title}\nECE = {rc.ece:.3f}")

    for ax in axes[n:]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, class_names, normalize: str = "row", title: str | None = None):
    """Row- (or column-) normalised confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize == "row":
        with np.errstate(invalid="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == "col":
        with np.errstate(invalid="ignore"):
            cm = cm / cm.sum(axis=0, keepdims=True)

    fig, ax = plt.subplots(figsize=(0.5 * len(class_names) + 3,
                                    0.5 * len(class_names) + 2.5))
    im = ax.imshow(np.nan_to_num(cm), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
