"""Aggregate nested donor-held-out robustness analyses on the Modal volume."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


HUMAN_DATASETS = (
    "TNBC_Breast_Cancer",
    "Indonesia_PBMC",
    "Brain_Atlas",
    "Multi_Tissue_TME",
    "Human_Pancreas",
)
REPRESENTATIONS = (
    "scgpt_embeddings",
    "matched_expression",
    "svd512_expression",
)
CLASSIFIERS = ("logreg", "random_forest", "xgboost")


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")


def _macro_f1_from_confusions(confusions):
    import numpy as np

    confusions = np.asarray(confusions, dtype=np.float64)
    diagonal = np.diagonal(confusions, axis1=-2, axis2=-1)
    denominator = confusions.sum(axis=-1) + confusions.sum(axis=-2)
    per_class = np.divide(
        2.0 * diagonal,
        denominator,
        out=np.zeros_like(diagonal),
        where=denominator > 0,
    )
    return per_class.mean(axis=-1)


def aggregate_v3_results(
    grouped_root: str,
    scarcity_root: str,
    pig_root: str,
    output_root: str,
    n_bootstrap: int = 5000,
) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support

    from revision_analysis_v3 import calculate_metrics

    grouped = Path(grouped_root)
    scarcity = Path(scarcity_root)
    pig = Path(pig_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    metrics_parts = []
    temperature_parts = []
    design_rows = []
    svd_rows = []
    for dataset in HUMAN_DATASETS:
        dataset_complete = []
        for fold in range(5):
            fold_root = grouped / dataset / f"fold_{fold}"
            complete_path = fold_root / "complete.json"
            if not complete_path.exists():
                raise FileNotFoundError(complete_path)
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            dataset_complete.append(complete)
            svd_audit = json.loads(
                (fold_root / "svd_audit.json").read_text(encoding="utf-8")
            )
            svd_rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    **svd_audit,
                }
            )
            metrics_parts.append(pd.read_csv(fold_root / "metrics.csv"))
            temperature_parts.append(
                pd.read_csv(fold_root / "temperature_scaling_metrics.csv")
            )
        design_rows.append(
            {
                "dataset": dataset,
                "group_column": dataset_complete[0]["group_column"],
                "n_unique_groups": dataset_complete[0]["n_unique_groups_total"],
                "n_outer_folds": 5,
                "n_classes": dataset_complete[0]["n_classes"],
                "n_oof_test_cells": int(sum(row["n_test_cells"] for row in dataset_complete)),
                "matched_hvgs_min": int(min(row["n_matched_hvgs"] for row in dataset_complete)),
                "matched_hvgs_max": int(max(row["n_matched_hvgs"] for row in dataset_complete)),
                "zero_input_cells_excluded": int(
                    sum(row["n_zero_input_test_excluded"] for row in dataset_complete)
                ),
            }
        )
    metrics = pd.concat(metrics_parts, ignore_index=True)
    temperature = pd.concat(temperature_parts, ignore_index=True)
    design = pd.DataFrame(design_rows)
    svd_audit = pd.DataFrame(svd_rows)
    metrics.to_csv(output / "human_fold_metrics.csv", index=False)
    temperature.to_csv(output / "temperature_scaling_fold_metrics.csv", index=False)
    design.to_csv(output / "human_dataset_design.csv", index=False)
    svd_audit.to_csv(output / "human_svd_audit.csv", index=False)

    fold_summary = (
        metrics.groupby(["dataset", "representation", "classifier"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            ece_mean=("ece_equal_width_10", "mean"),
            brier_mean=("multiclass_brier", "mean"),
            nll_mean=("negative_log_likelihood", "mean"),
            n_test_cells=("n_test_cells", "sum"),
        )
    )
    fold_summary.to_csv(output / "human_fold_summary.csv", index=False)

    parameter_rows = []
    for keys, subset in metrics.groupby(["dataset", "representation", "classifier"]):
        for parameters, frequency in Counter(subset["selected_parameters"]).items():
            parameter_rows.append(
                {
                    "dataset": keys[0],
                    "representation": keys[1],
                    "classifier": keys[2],
                    "selected_parameters": parameters,
                    "fold_frequency": int(frequency),
                }
            )
    pd.DataFrame(parameter_rows).to_csv(
        output / "selected_hyperparameter_frequencies.csv", index=False
    )

    pooled_rows = []
    per_class_rows = []
    confusion_rows = []
    reliability_rows = []
    comparison_rows = []
    for dataset_index, dataset in enumerate(HUMAN_DATASETS):
        predictions = {}
        reference_indices = None
        reference_y = None
        reference_groups = None
        reference_classes = None
        for representation in REPRESENTATIONS:
            for classifier in CLASSIFIERS:
                model = f"{representation} plus {classifier}"
                parts = []
                for fold in range(5):
                    path = (
                        grouped
                        / dataset
                        / f"fold_{fold}"
                        / f"predictions__{_slug(model)}.npz"
                    )
                    values = np.load(path, allow_pickle=False)
                    parts.append(
                        {
                            "test_index": values["test_index"].astype(np.int64),
                            "group_id": values["group_id"].astype(str),
                            "y_true": values["y_true"].astype(np.int64),
                            "y_prob": values["y_prob"].astype(np.float32),
                            "class_names": values["class_names"].astype(str),
                        }
                    )
                indices = np.concatenate([part["test_index"] for part in parts])
                if len(np.unique(indices)) != len(indices):
                    raise AssertionError(f"Repeated OOF indices for {dataset} {model}")
                order = np.argsort(indices, kind="mergesort")
                groups = np.concatenate([part["group_id"] for part in parts])[order]
                y_true = np.concatenate([part["y_true"] for part in parts])[order]
                y_prob = np.concatenate([part["y_prob"] for part in parts])[order]
                class_names = parts[0]["class_names"]
                if reference_indices is None:
                    reference_indices = indices[order]
                    reference_y = y_true
                    reference_groups = groups
                    reference_classes = class_names
                elif not (
                    np.array_equal(reference_indices, indices[order])
                    and np.array_equal(reference_y, y_true)
                    and np.array_equal(reference_groups, groups)
                    and np.array_equal(reference_classes, class_names)
                ):
                    raise AssertionError(f"OOF prediction alignment failed for {dataset}")
                predictions[(representation, classifier)] = y_prob.argmax(axis=1)
                pooled_rows.append(
                    {
                        "dataset": dataset,
                        "representation": representation,
                        "classifier": classifier,
                        "n_cells": len(y_true),
                        "n_groups": int(len(np.unique(groups))),
                        **calculate_metrics(y_true, y_prob),
                    }
                )
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true,
                    y_prob.argmax(axis=1),
                    labels=np.arange(len(class_names)),
                    zero_division=0,
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
                np.add.at(confusion, (y_true, y_prob.argmax(axis=1)), 1)
                row_total = confusion.sum(axis=1, keepdims=True)
                normalized = np.divide(
                    confusion,
                    row_total,
                    out=np.zeros_like(confusion, dtype=np.float64),
                    where=row_total > 0,
                )
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
                correct = y_prob.argmax(axis=1) == y_true
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

        unique_groups = np.unique(reference_groups)
        group_lookup = {group: index for index, group in enumerate(unique_groups)}
        group_code = np.asarray([group_lookup[value] for value in reference_groups])
        rng = np.random.default_rng(20261311 + dataset_index)
        group_weights = rng.multinomial(
            len(unique_groups),
            np.repeat(1.0 / len(unique_groups), len(unique_groups)),
            size=n_bootstrap,
        ).astype(np.float64)
        boot_f1 = {}
        point_f1 = {}
        for key, prediction in predictions.items():
            confusion_by_group = np.zeros(
                (len(unique_groups), len(reference_classes), len(reference_classes)),
                dtype=np.int64,
            )
            np.add.at(confusion_by_group, (group_code, reference_y, prediction), 1)
            draws = (group_weights @ confusion_by_group.reshape(len(unique_groups), -1)).reshape(
                n_bootstrap, len(reference_classes), len(reference_classes)
            )
            boot_f1[key] = _macro_f1_from_confusions(draws)
            point_f1[key] = float(_macro_f1_from_confusions(confusion_by_group.sum(axis=0)))
        for classifier in CLASSIFIERS:
            for representation_a, representation_b, comparison in (
                ("matched_expression", "scgpt_embeddings", "matched_expression_minus_scgpt"),
                ("matched_expression", "svd512_expression", "matched_expression_minus_svd512"),
                ("svd512_expression", "scgpt_embeddings", "svd512_minus_scgpt"),
            ):
                key_a = (representation_a, classifier)
                key_b = (representation_b, classifier)
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

    pooled = pd.DataFrame(pooled_rows)
    per_class = pd.DataFrame(per_class_rows)
    confusion = pd.DataFrame(confusion_rows)
    reliability = pd.DataFrame(reliability_rows)
    comparisons = pd.DataFrame(comparison_rows)
    pooled.to_csv(output / "human_pooled_metrics.csv", index=False)
    per_class.to_csv(output / "human_per_class_metrics.csv", index=False)
    confusion.to_csv(output / "human_confusion_matrices_long.csv", index=False)
    reliability.to_csv(output / "human_reliability_bins.csv", index=False)
    comparisons.to_csv(output / "human_group_bootstrap_comparisons.csv", index=False)

    temperature_summary = (
        temperature.groupby(
            ["dataset", "representation", "classifier", "calibrated"],
            as_index=False,
        )
        .agg(
            temperature_mean=("temperature", "mean"),
            temperature_sd=("temperature", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            ece_mean=("ece_equal_width_10", "mean"),
            ece_sd=("ece_equal_width_10", "std"),
            brier_mean=("multiclass_brier", "mean"),
            nll_mean=("negative_log_likelihood", "mean"),
            n_test_cells=("n_test_cells", "sum"),
        )
    )
    temperature_summary.to_csv(output / "temperature_scaling_summary.csv", index=False)

    scarcity_parts = []
    for fold in range(5):
        path = scarcity / f"fold_{fold}" / "strict_scarcity_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        scarcity_parts.append(pd.read_csv(path))
    scarcity_metrics = pd.concat(scarcity_parts, ignore_index=True)
    scarcity_metrics.to_csv(output / "strict_scarcity_metrics.csv", index=False)
    scarcity_fold = (
        scarcity_metrics.groupby(
            ["fold", "labels_per_class", "representation"], as_index=False
        )
        .agg(
            macro_f1=("macro_f1", "mean"),
            ece=("ece_equal_width_10", "mean"),
            brier=("multiclass_brier", "mean"),
            nll=("negative_log_likelihood", "mean"),
            n_matched_hvgs=("n_matched_hvgs", "mean"),
        )
    )
    scarcity_summary = (
        scarcity_fold.groupby(["labels_per_class", "representation"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            ece_mean=("ece", "mean"),
            ece_sd=("ece", "std"),
            brier_mean=("brier", "mean"),
            nll_mean=("nll", "mean"),
            matched_hvgs_mean=("n_matched_hvgs", "mean"),
        )
    )
    scarcity_summary.to_csv(output / "strict_scarcity_summary.csv", index=False)
    scarcity_pivot = scarcity_fold.pivot(
        index=["fold", "labels_per_class"],
        columns="representation",
        values="macro_f1",
    ).reset_index()
    scarcity_pivot["matched_minus_scgpt"] = (
        scarcity_pivot["matched_expression"] - scarcity_pivot["scgpt_embeddings"]
    )
    scarcity_difference = (
        scarcity_pivot.groupby("labels_per_class", as_index=False)
        .agg(
            difference_mean=("matched_minus_scgpt", "mean"),
            difference_sd=("matched_minus_scgpt", "std"),
            difference_min=("matched_minus_scgpt", "min"),
            difference_max=("matched_minus_scgpt", "max"),
        )
    )
    scarcity_difference.to_csv(output / "strict_scarcity_differences.csv", index=False)

    pig_metrics = pd.read_csv(pig / "metrics.csv")
    pig_metrics.to_csv(output / "pig_cross_species_metrics.csv", index=False)
    pig_per_class = []
    pig_confusion = []
    for row in pig_metrics.itertuples(index=False):
        model = f"{row.representation} plus {row.classifier}"
        values = np.load(pig / f"predictions__{_slug(model)}.npz", allow_pickle=False)
        y_true = values["y_true"].astype(np.int64)
        y_prob = values["y_prob"].astype(np.float64)
        class_names = values["class_names"].astype(str)
        prediction = y_prob.argmax(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            prediction,
            labels=np.arange(len(class_names)),
            zero_division=0,
        )
        counts = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        np.add.at(counts, (y_true, prediction), 1)
        for index, class_name in enumerate(class_names):
            pig_per_class.append(
                {
                    "representation": row.representation,
                    "classifier": row.classifier,
                    "class_name": class_name,
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
            )
        for true_index, true_name in enumerate(class_names):
            for predicted_index, predicted_name in enumerate(class_names):
                pig_confusion.append(
                    {
                        "representation": row.representation,
                        "classifier": row.classifier,
                        "true_class": true_name,
                        "predicted_class": predicted_name,
                        "count": int(counts[true_index, predicted_index]),
                    }
                )
    pd.DataFrame(pig_per_class).to_csv(
        output / "pig_cross_species_per_class_metrics.csv", index=False
    )
    pd.DataFrame(pig_confusion).to_csv(
        output / "pig_cross_species_confusion_long.csv", index=False
    )

    payload = {
        "human_datasets": list(HUMAN_DATASETS),
        "n_human_datasets": len(HUMAN_DATASETS),
        "n_outer_folds_per_dataset": 5,
        "n_primary_classifier_matched_comparisons": 15,
        "n_bootstrap": int(n_bootstrap),
        "n_human_oof_cells_total": int(design["n_oof_test_cells"].sum()),
        "strict_scarcity_rows": int(len(scarcity_metrics)),
        "pig_analysis_role": "cross-species stress test",
    }
    (output / "aggregate_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload
