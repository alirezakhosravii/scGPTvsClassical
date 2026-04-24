"""scgpt_vs_classical: calibration-aware benchmarking of scGPT vs classical ML.

Modules
-------
calibration
    Expected Calibration Error (ECE), reliability diagrams, and non-parametric
    bootstrap confidence intervals for Macro F1.
scgpt_pipeline
    HVG selection with the scGPT recipe, vocabulary matching, and frozen
    embedding extraction from the pre-trained scGPT whole-human checkpoint.
classical
    XGBoost / Random Forest / Logistic Regression trainers used as baselines
    on highly variable genes.
finetune
    End-to-end fine-tuning of scGPT with a classification head on a single
    target dataset (the experiment underpinning paper Table 3).
plotting
    Per-dataset 6-panel reliability diagram grids and per-class confusion
    matrices used in the paper figures and supplementary material.
"""

__version__ = "1.0.0"
__all__ = [
    "calibration",
    "scgpt_pipeline",
    "classical",
    "finetune",
    "plotting",
]
