#!/usr/bin/env python3
"""Print the exact statements needed to finalize the manuscript text."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    root = args.results

    design = pd.read_csv(root / "human_dataset_design.csv")
    pooled = pd.read_csv(root / "human_pooled_metrics.csv")
    comparisons = pd.read_csv(root / "human_group_bootstrap_comparisons.csv")
    temperature = pd.read_csv(root / "temperature_scaling_fold_metrics.csv")
    per_class = pd.read_csv(root / "human_per_class_metrics.csv")
    confusion = pd.read_csv(root / "human_confusion_matrices_long.csv")

    print("DESIGN")
    print(design.to_string(index=False))
    print("\nPOOLED MACRO F1")
    print(pooled.pivot_table(index=["dataset", "classifier"], columns="representation", values="macro_f1").round(6).to_string())

    primary = comparisons[comparisons.comparison == "matched_expression_minus_scgpt"]
    print("\nPRIMARY COUNTS")
    print("expression observed wins", int((primary.difference_a_minus_b > 0).sum()), "of", len(primary))
    print("CIs entirely > 0", int((primary.group_bootstrap_ci_low > 0).sum()))
    print("CIs crossing 0", int(((primary.group_bootstrap_ci_low <= 0) & (primary.group_bootstrap_ci_high >= 0)).sum()))
    print(primary[["dataset", "classifier", "difference_a_minus_b", "group_bootstrap_ci_low", "group_bootstrap_ci_high", "bootstrap_probability_gt_zero"]].to_string(index=False))

    print("\nALL COMPARISON COUNTS")
    for name, subset in comparisons.groupby("comparison"):
        print(name, "positive", int((subset.difference_a_minus_b > 0).sum()), "CI positive", int((subset.group_bootstrap_ci_low > 0).sum()), "of", len(subset))

    temp = temperature.copy()
    if temp.calibrated.dtype == object:
        temp.calibrated = temp.calibrated.astype(str).str.lower().eq("true")
    else:
        temp.calibrated = temp.calibrated.astype(bool)
    keys = ["dataset", "fold", "representation", "classifier"]
    before = temp[~temp.calibrated].set_index(keys)
    after = temp[temp.calibrated].set_index(keys)
    for value in ("ece_equal_width_10", "multiclass_brier", "negative_log_likelihood"):
        difference = after[value] - before[value]
        print(f"\nTEMPERATURE {value}: mean before={before[value].mean():.6f}, after={after[value].mean():.6f}, improved={int((difference < 0).sum())}/{len(difference)}, worsened={int((difference > 0).sum())}")

    for dataset in ("Multi_Tissue_TME", "Human_Pancreas"):
        print(f"\nPER CLASS {dataset} XGBOOST")
        sub = per_class[(per_class.dataset == dataset) & (per_class.classifier == "xgboost")]
        print(sub[["representation", "class_name", "support", "precision", "recall", "f1"]].sort_values(["representation", "f1"]).to_string(index=False))
        for representation in ("scgpt_embeddings", "matched_expression"):
            c = confusion[(confusion.dataset == dataset) & (confusion.classifier == "xgboost") & (confusion.representation == representation) & (confusion.true_class != confusion.predicted_class)]
            print("top confusions", representation)
            print(c.nlargest(8, "row_proportion")[["true_class", "predicted_class", "count", "row_proportion"]].to_string(index=False))


if __name__ == "__main__":
    main()
