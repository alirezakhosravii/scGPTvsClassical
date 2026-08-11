"""Aggregate completed, durably stored outputs without launching new compute.

The primary paired benchmark is restricted to pipelines with five complete
donor-held-out prediction folds.  This deliberately ignores partial TME and
pancreas jobs and never treats a partly generated pipeline as evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from revision_analysis_v3 import calculate_metrics


HUMAN_DATASETS = ("TNBC_Breast_Cancer", "Indonesia_PBMC", "Brain_Atlas")
REPRESENTATIONS = ("scgpt_embeddings", "matched_expression", "svd512_expression")
CLASSIFIERS = ("logreg", "random_forest", "xgboost")


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")


def _prediction_path(root: Path, dataset: str, fold: int, representation: str, classifier: str) -> Path:
    model = f"{representation} plus {classifier}"
    return root / dataset / f"fold_{fold}" / f"predictions__{_slug(model)}.npz"


def _pipeline_complete(root: Path, dataset: str, representation: str, classifier: str) -> bool:
    return all(_prediction_path(root, dataset, fold, representation, classifier).exists() for fold in range(5))


def _macro_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    confusions = np.asarray(confusions, dtype=np.float64)
    diagonal = np.diagonal(confusions, axis1=-2, axis2=-1)
    denominator = confusions.sum(axis=-1) + confusions.sum(axis=-2)
    per_class = np.divide(2.0 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0)
    return per_class.mean(axis=-1)


def _load_pipeline(root: Path, dataset: str, representation: str, classifier: str):
    parts = []
    fold_rows = []
    for fold in range(5):
        path = _prediction_path(root, dataset, fold, representation, classifier)
        values = np.load(path, allow_pickle=False)
        test_index = values["test_index"].astype(np.int64)
        groups = values["group_id"].astype(str)
        y_true = values["y_true"].astype(np.int64)
        y_prob = values["y_prob"].astype(np.float32)
        class_names = values["class_names"].astype(str)
        fold_rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "representation": representation,
                "classifier": classifier,
                "n_test_groups": int(len(np.unique(groups))),
                "n_test_cells": int(len(y_true)),
                **calculate_metrics(y_true, y_prob),
            }
        )
        parts.append((test_index, groups, y_true, y_prob, class_names))
    indices = np.concatenate([part[0] for part in parts])
    if len(np.unique(indices)) != len(indices):
        raise AssertionError(f"Repeated OOF indices for {dataset} {representation} {classifier}")
    order = np.argsort(indices, kind="mergesort")
    groups = np.concatenate([part[1] for part in parts])[order]
    y_true = np.concatenate([part[2] for part in parts])[order]
    y_prob = np.concatenate([part[3] for part in parts])[order]
    class_names = parts[0][4]
    return indices[order], groups, y_true, y_prob, class_names, fold_rows


def aggregate_existing_results(grouped_root: str, scarcity_root: str, pig_root: str, output_root: str, n_bootstrap: int = 5000) -> dict:
    grouped = Path(grouped_root)
    scarcity = Path(scarcity_root)
    pig = Path(pig_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    pooled_rows = []
    per_class_rows = []
    confusion_rows = []
    reliability_rows = []
    comparison_rows = []
    design_rows = []
    svd_rows = []
    coverage_rows = []

    for dataset_index, dataset in enumerate(HUMAN_DATASETS):
        available = {}
        reference = None
        for representation in REPRESENTATIONS:
            for classifier in CLASSIFIERS:
                complete = _pipeline_complete(grouped, dataset, representation, classifier)
                coverage_rows.append(
                    {
                        "dataset": dataset,
                        "representation": representation,
                        "classifier": classifier,
                        "five_oof_folds_available": complete,
                        "included": complete,
                    }
                )
                if not complete:
                    continue
                indices, groups, y_true, y_prob, class_names, pipeline_fold_rows = _load_pipeline(
                    grouped, dataset, representation, classifier
                )
                fold_rows.extend(pipeline_fold_rows)
                if reference is None:
                    reference = (indices, groups, y_true, class_names)
                else:
                    ref_indices, ref_groups, ref_y, ref_classes = reference
                    if not (
                        np.array_equal(ref_indices, indices)
                        and np.array_equal(ref_groups, groups)
                        and np.array_equal(ref_y, y_true)
                        and np.array_equal(ref_classes, class_names)
                    ):
                        raise AssertionError(f"OOF alignment failed for {dataset}")
                prediction = y_prob.argmax(axis=1)
                available[(representation, classifier)] = prediction
                pooled_rows.append(
                    {
                        "dataset": dataset,
                        "representation": representation,
                        "classifier": classifier,
                        "n_cells": int(len(y_true)),
                        "n_groups": int(len(np.unique(groups))),
                        **calculate_metrics(y_true, y_prob),
                    }
                )
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, prediction, labels=np.arange(len(class_names)), zero_division=0
                )
                for class_index, class_name in enumerate(class_names):
                    per_class_rows.append(
                        {
                            "dataset": dataset,
                            "representation": representation,
                            "classifier": classifier,
                            "class_index": class_index,
                            "class_name": class_name,
                            "precision": float(precision[class_index]),
                            "recall": float(recall[class_index]),
                            "f1": float(f1[class_index]),
                            "support": int(support[class_index]),
                        }
                    )
                confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
                np.add.at(confusion, (y_true, prediction), 1)
                totals = confusion.sum(axis=1, keepdims=True)
                normalized = np.divide(confusion, totals, out=np.zeros_like(confusion, dtype=float), where=totals > 0)
                for true_index, true_name in enumerate(class_names):
                    for predicted_index, predicted_name in enumerate(class_names):
                        confusion_rows.append(
                            {
                                "dataset": dataset,
                                "representation": representation,
                                "classifier": classifier,
                                "true_class": true_name,
                                "predicted_class": predicted_name,
                                "count": int(confusion[true_index, predicted_index]),
                                "row_proportion": float(normalized[true_index, predicted_index]),
                            }
                        )
                confidence = y_prob.max(axis=1)
                correct = prediction == y_true
                bin_index = np.minimum((confidence * 10).astype(int), 9)
                for bin_number in range(10):
                    mask = bin_index == bin_number
                    reliability_rows.append(
                        {
                            "dataset": dataset,
                            "representation": representation,
                            "classifier": classifier,
                            "bin": bin_number + 1,
                            "lower_edge": bin_number / 10,
                            "upper_edge": (bin_number + 1) / 10,
                            "n": int(mask.sum()),
                            "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                            "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
                        }
                    )

        if reference is None:
            raise RuntimeError(f"No complete prediction pipeline for {dataset}")
        ref_indices, ref_groups, ref_y, ref_classes = reference
        selected_counts = []
        for fold in range(5):
            selected = json.loads((grouped / dataset / f"fold_{fold}" / "selected_genes.json").read_text())
            selected_counts.append(int(selected["n_hvg_matched"]))
            svd_path = grouped / dataset / f"fold_{fold}" / "svd_audit.json"
            if svd_path.exists():
                svd_rows.append({"dataset": dataset, "fold": fold, **json.loads(svd_path.read_text())})
        design_rows.append(
            {
                "dataset": dataset,
                "group_column": "donor_id",
                "n_unique_groups": int(len(np.unique(ref_groups))),
                "n_outer_folds": 5,
                "n_classes": int(len(ref_classes)),
                "n_oof_test_cells": int(len(ref_y)),
                "matched_hvgs_min": int(min(selected_counts)),
                "matched_hvgs_max": int(max(selected_counts)),
            }
        )

        unique_groups = np.unique(ref_groups)
        lookup = {group: index for index, group in enumerate(unique_groups)}
        group_code = np.asarray([lookup[value] for value in ref_groups])
        rng = np.random.default_rng(20261311 + dataset_index)
        group_weights = rng.multinomial(
            len(unique_groups), np.repeat(1.0 / len(unique_groups), len(unique_groups)), size=n_bootstrap
        ).astype(float)
        boot_f1 = {}
        point_f1 = {}
        for key, prediction in available.items():
            by_group = np.zeros((len(unique_groups), len(ref_classes), len(ref_classes)), dtype=np.int64)
            np.add.at(by_group, (group_code, ref_y, prediction), 1)
            draws = (group_weights @ by_group.reshape(len(unique_groups), -1)).reshape(
                n_bootstrap, len(ref_classes), len(ref_classes)
            )
            boot_f1[key] = _macro_f1_from_confusions(draws)
            point_f1[key] = float(_macro_f1_from_confusions(by_group.sum(axis=0)))
        contrasts = (
            ("matched_expression", "scgpt_embeddings", "matched_expression_minus_scgpt"),
            ("matched_expression", "svd512_expression", "matched_expression_minus_svd512"),
            ("svd512_expression", "scgpt_embeddings", "svd512_minus_scgpt"),
        )
        for classifier in CLASSIFIERS:
            for representation_a, representation_b, comparison in contrasts:
                key_a = (representation_a, classifier)
                key_b = (representation_b, classifier)
                if key_a not in available or key_b not in available:
                    continue
                difference = boot_f1[key_a] - boot_f1[key_b]
                comparison_rows.append(
                    {
                        "dataset": dataset,
                        "classifier": classifier,
                        "comparison": comparison,
                        "representation_a": representation_a,
                        "representation_b": representation_b,
                        "macro_f1_a": point_f1[key_a],
                        "macro_f1_b": point_f1[key_b],
                        "difference_a_minus_b": point_f1[key_a] - point_f1[key_b],
                        "group_bootstrap_ci_low": float(np.quantile(difference, 0.025)),
                        "group_bootstrap_ci_high": float(np.quantile(difference, 0.975)),
                        "bootstrap_probability_gt_zero": float(np.mean(difference > 0)),
                        "n_groups": int(len(unique_groups)),
                        "n_bootstrap": int(n_bootstrap),
                    }
                )

    fold_metrics = pd.DataFrame(fold_rows)
    pooled = pd.DataFrame(pooled_rows)
    comparisons = pd.DataFrame(comparison_rows)
    fold_metrics.to_csv(output / "human_fold_metrics.csv", index=False)
    pooled.to_csv(output / "human_pooled_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(output / "human_per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output / "human_confusion_matrices_long.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(output / "human_reliability_bins.csv", index=False)
    comparisons.to_csv(output / "human_group_bootstrap_comparisons.csv", index=False)
    pd.DataFrame(design_rows).to_csv(output / "human_dataset_design.csv", index=False)
    pd.DataFrame(svd_rows).to_csv(output / "human_svd_audit.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "human_pipeline_coverage.csv", index=False)
    fold_summary = fold_metrics.groupby(["dataset", "representation", "classifier"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        macro_f1_min=("macro_f1", "min"), macro_f1_max=("macro_f1", "max"),
        accuracy_mean=("accuracy", "mean"), balanced_accuracy_mean=("balanced_accuracy", "mean"),
        ece_mean=("ece_equal_width_10", "mean"), brier_mean=("multiclass_brier", "mean"),
        nll_mean=("negative_log_likelihood", "mean"), n_test_cells=("n_test_cells", "sum")
    )
    fold_summary.to_csv(output / "human_fold_summary.csv", index=False)

    temperature_parts = []
    parameter_rows = []
    for dataset in HUMAN_DATASETS:
        for fold in range(5):
            fold_root = grouped / dataset / f"fold_{fold}"
            temperature_path = fold_root / "temperature_scaling_metrics.csv"
            if temperature_path.exists():
                temperature_parts.append(pd.read_csv(temperature_path))
            metrics_path = fold_root / "metrics.csv"
            if metrics_path.exists():
                for row in pd.read_csv(metrics_path).itertuples(index=False):
                    parameter_rows.append(
                        {
                            "dataset": dataset,
                            "representation": row.representation,
                            "classifier": row.classifier,
                            "selected_parameters": row.selected_parameters,
                        }
                    )
    temperature = pd.concat(temperature_parts, ignore_index=True)
    temperature.to_csv(output / "temperature_scaling_fold_metrics.csv", index=False)
    temperature.groupby(["dataset", "representation", "classifier", "calibrated"], as_index=False).agg(
        temperature_mean=("temperature", "mean"), temperature_sd=("temperature", "std"),
        macro_f1_mean=("macro_f1", "mean"), ece_mean=("ece_equal_width_10", "mean"),
        ece_sd=("ece_equal_width_10", "std"), brier_mean=("multiclass_brier", "mean"),
        nll_mean=("negative_log_likelihood", "mean"), n_test_cells=("n_test_cells", "sum")
    ).to_csv(output / "temperature_scaling_summary.csv", index=False)
    if parameter_rows:
        params = pd.DataFrame(parameter_rows)
        frequencies = []
        for keys, subset in params.groupby(["dataset", "representation", "classifier"]):
            for value, count in Counter(subset["selected_parameters"]).items():
                frequencies.append(
                    {"dataset": keys[0], "representation": keys[1], "classifier": keys[2],
                     "selected_parameters": value, "fold_frequency": int(count)}
                )
        pd.DataFrame(frequencies).to_csv(output / "selected_hyperparameter_frequencies.csv", index=False)

    scarcity_parts = [pd.read_csv(scarcity / f"fold_{fold}" / "strict_scarcity_metrics.csv") for fold in range(5)]
    scarcity_metrics = pd.concat(scarcity_parts, ignore_index=True)
    scarcity_metrics.to_csv(output / "strict_scarcity_metrics.csv", index=False)
    scarcity_fold = scarcity_metrics.groupby(["fold", "labels_per_class", "representation"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"), ece=("ece_equal_width_10", "mean"),
        brier=("multiclass_brier", "mean"), nll=("negative_log_likelihood", "mean"),
        n_matched_hvgs=("n_matched_hvgs", "mean")
    )
    scarcity_summary = scarcity_fold.groupby(["labels_per_class", "representation"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        ece_mean=("ece", "mean"), ece_sd=("ece", "std"), brier_mean=("brier", "mean"),
        nll_mean=("nll", "mean"), matched_hvgs_mean=("n_matched_hvgs", "mean")
    )
    scarcity_summary.to_csv(output / "strict_scarcity_summary.csv", index=False)
    pivot = scarcity_fold.pivot(index=["fold", "labels_per_class"], columns="representation", values="macro_f1").reset_index()
    pivot["matched_minus_scgpt"] = pivot["matched_expression"] - pivot["scgpt_embeddings"]
    pivot.groupby("labels_per_class", as_index=False).agg(
        difference_mean=("matched_minus_scgpt", "mean"), difference_sd=("matched_minus_scgpt", "std"),
        difference_min=("matched_minus_scgpt", "min"), difference_max=("matched_minus_scgpt", "max")
    ).to_csv(output / "strict_scarcity_differences.csv", index=False)

    pig_metrics = pd.read_csv(pig / "metrics.csv")
    pig_metrics.to_csv(output / "pig_cross_species_metrics.csv", index=False)
    payload = {
        "human_datasets": list(HUMAN_DATASETS),
        "n_human_datasets": len(HUMAN_DATASETS),
        "n_outer_folds_per_dataset": 5,
        "n_primary_expression_scgpt_comparisons": 9,
        "n_bootstrap": int(n_bootstrap),
        "n_human_oof_cells_total": int(pd.DataFrame(design_rows)["n_oof_test_cells"].sum()),
        "strict_scarcity_rows": int(len(scarcity_metrics)),
        "pig_analysis_role": "cross-species stress test",
        "compute_policy": "existing completed predictions only; no additional GPU jobs",
    }
    (output / "aggregate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("grouped_root")
    parser.add_argument("scarcity_root")
    parser.add_argument("pig_root")
    parser.add_argument("output_root")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(aggregate_existing_results(args.grouped_root, args.scarcity_root, args.pig_root, args.output_root, args.n_bootstrap), indent=2))
