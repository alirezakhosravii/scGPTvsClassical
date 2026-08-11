#!/usr/bin/env python3
"""Build publication figures and LaTeX tables from the final aggregate CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


DATASET_ORDER = [
    "TNBC_Breast_Cancer",
    "Indonesia_PBMC",
    "Brain_Atlas",
]
DATASET_LABEL = {
    "TNBC_Breast_Cancer": "TNBC",
    "Indonesia_PBMC": "PBMC",
    "Brain_Atlas": "Brain",
}
REP_ORDER = ["scgpt_embeddings", "matched_expression", "svd512_expression"]
REP_LABEL = {
    "scgpt_embeddings": "scGPT embeddings",
    "matched_expression": "Matched expression",
    "svd512_expression": "SVD-512",
}
CLF_ORDER = ["logreg", "random_forest", "xgboost"]
CLF_LABEL = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
COLORS = {
    "scgpt_embeddings": "#3561A7",
    "matched_expression": "#D55E00",
    "svd512_expression": "#008B72",
}
INK = "#202428"
MID = "#697078"
GRID = "#D7DBDE"
LIGHT = "#F4F5F6"


def tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def metric(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=350,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def configure_style() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#AEB4B9",
            "axes.titlecolor": INK,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
        }
    )


def figure_design(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.1)
    ax.axis("off")

    panel_specs = [(0.15, 3.45), (3.75, 5.05), (8.95, 2.90)]
    for x, width in panel_specs:
        ax.add_patch(Rectangle((x, 0.25), width, 6.55, facecolor="white", edgecolor=GRID, linewidth=0.9))

    def panel_heading(x: float, letter: str, heading: str) -> None:
        ax.text(x, 6.55, letter, fontsize=15, weight="bold", va="center")
        ax.text(x + 0.34, 6.55, heading, fontsize=10.2, weight="bold", va="center", color=MID)

    panel_heading(0.38, "A", "DATA AND BIOLOGICAL HOLDOUT")
    panel_heading(3.98, "B", "REPRESENTATION-MATCHED LEARNING")
    panel_heading(9.18, "C", "PRESPECIFIED EVALUATION")

    # Panel A: exact analysed cohort and the grouped outer-fold structure.
    columns = [0.42, 1.56, 2.32, 2.95]
    headers = ["Atlas", "Cells", "Donors", "Labels"]
    for x, label in zip(columns, headers):
        ax.text(x, 5.98, label, fontsize=8.4, weight="bold", color=MID, ha="left")
    rows = [
        ("TNBC", "427,813", "101", "7"),
        ("PBMC", "462,034", "199", "13"),
        ("Brain", "75,580", "12", "10"),
    ]
    for row_index, row in enumerate(rows):
        y = 5.50 - row_index * 0.53
        ax.plot([0.40, 3.32], [y - 0.20, y - 0.20], color="#ECEEEF", lw=0.7)
        for x, value in zip(columns, row):
            ax.text(x, y, value, fontsize=9.2, ha="left", va="center", weight="bold" if x == columns[0] else "normal")
    ax.text(0.42, 3.80, "965,427 analysed cells  |  312 donors", fontsize=9.2, weight="bold")
    ax.text(0.42, 3.49, "Five outer folds per atlas", fontsize=9.0, color=MID)

    fold_x, fold_y, fold_w, fold_h = 0.50, 1.34, 2.58, 0.28
    test_color = "#7259A6"
    for fold in range(5):
        y = fold_y + (4 - fold) * 0.37
        ax.text(0.42, y + fold_h / 2, str(fold + 1), fontsize=7.8, ha="right", va="center", color=MID)
        for part in range(5):
            color = test_color if part == fold else "#D8DCE0"
            ax.add_patch(Rectangle((fold_x + part * fold_w / 5, y), fold_w / 5 - 0.025, fold_h, facecolor=color, edgecolor="white", linewidth=0.4))
    ax.text(fold_x, 1.05, "training donors", fontsize=7.8, color=MID, ha="left")
    ax.text(fold_x + fold_w, 1.05, "held-out donors", fontsize=7.8, color=test_color, ha="right")
    ax.text(0.42, 0.56, "Each donor is tested once; train/test donors never overlap.", fontsize=8.4)

    # Panel B: a single train-only gene space branches into the three representations.
    ax.add_patch(Rectangle((4.02, 5.45), 4.50, 0.63, facecolor=LIGHT, edgecolor=GRID, linewidth=0.8))
    ax.text(6.27, 5.83, "Outer-training donors only", ha="center", va="center", fontsize=9.3, weight="bold")
    ax.text(6.27, 5.58, "HVG selection + scGPT vocabulary matching", ha="center", va="center", fontsize=8.3, color=MID)

    rep_cards = [
        (4.00, "Frozen scGPT", "512-dimensional\nweights fixed", "scgpt_embeddings"),
        (5.62, "Matched expression", "930--1,172 genes\nlog-normalised", "matched_expression"),
        (7.24, "SVD-512", "fit on training\ndonors only", "svd512_expression"),
    ]
    for x, title, body, representation in rep_cards:
        color = COLORS[representation]
        ax.add_patch(Rectangle((x, 3.67), 1.42, 1.18, facecolor="white", edgecolor=GRID, linewidth=0.9))
        ax.add_patch(Rectangle((x, 4.76), 1.42, 0.09, facecolor=color, edgecolor=color, linewidth=0))
        ax.text(x + 0.71, 4.47, title, ha="center", va="center", fontsize=8.7, weight="bold", color=color)
        ax.text(x + 0.71, 4.05, body, ha="center", va="center", fontsize=7.9, linespacing=1.25)
        ax.add_patch(FancyArrowPatch((6.27, 5.45), (x + 0.71, 4.87), arrowstyle="-|>", mutation_scale=8, color="#8B9196", linewidth=0.9))

    ax.add_patch(Rectangle((4.21, 1.48), 4.12, 1.25, facecolor=LIGHT, edgecolor=GRID, linewidth=0.8))
    ax.text(6.27, 2.42, "Separately tuned within training donors", ha="center", va="center", fontsize=9.1, weight="bold")
    ax.text(6.27, 2.02, "Logistic Regression    Random Forest    XGBoost", ha="center", va="center", fontsize=8.4)
    ax.text(6.27, 1.70, "same classifier retained for every primary contrast", ha="center", va="center", fontsize=7.8, color=MID)
    for x, *_ in rep_cards:
        ax.add_patch(FancyArrowPatch((x + 0.71, 3.67), (6.27, 2.73), arrowstyle="-|>", mutation_scale=8, color="#8B9196", linewidth=0.9))
    ax.text(6.27, 0.75, "Locked outer-test predictions generated once", ha="center", va="center", fontsize=8.5, color="#7259A6", weight="bold")

    # Panel C: outcomes grouped by inferential role.
    outcomes = [
        ("PRIMARY", "Macro F1\nfixed-classifier contrasts"),
        ("UNCERTAINTY", "paired donor bootstrap\n5,000 resamples"),
        ("PROBABILITIES", "ECE, Brier, NLL\ntemperature scaling"),
        ("BIOLOGY", "class F1 + confusions\nstrict label scarcity"),
    ]
    y = 5.90
    for heading, body in outcomes:
        ax.text(9.25, y, heading, fontsize=7.6, weight="bold", color="#7259A6", va="top")
        ax.text(9.25, y - 0.26, body, fontsize=8.8, va="top", linespacing=1.25)
        y -= 1.28
    ax.plot([9.24, 11.55], [0.98, 0.98], color=GRID, lw=0.8)
    ax.text(9.25, 0.73, "SEPARATE STRESS TEST", fontsize=7.6, weight="bold", color=MID)
    ax.text(9.25, 0.48, "pig pancreas; random cell split", fontsize=8.5)

    fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.01)
    save_figure(fig, output / "figure_1_study_design")


def figure_performance(pooled: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.25), sharex=True, sharey=True)
    y = np.arange(len(CLF_ORDER))
    subtitles = {
        "TNBC_Breast_Cancer": "427,813 cells · 101 donors",
        "Indonesia_PBMC": "462,034 cells · 199 donors",
        "Brain_Atlas": "75,580 cells · 12 donors",
    }
    for panel, (ax, dataset) in enumerate(zip(axes, DATASET_ORDER)):
        subset = pooled[pooled["dataset"] == dataset]
        for row_index, classifier in enumerate(CLF_ORDER):
            row = subset[subset["classifier"] == classifier].set_index("representation")
            scgpt = float(row.loc["scgpt_embeddings", "macro_f1"])
            expression = float(row.loc["matched_expression", "macro_f1"])
            ax.plot([scgpt, expression], [row_index, row_index], color="#BFC4C8", lw=2.0, zorder=1)
            ax.scatter(scgpt, row_index, s=62, color=COLORS["scgpt_embeddings"], edgecolor="white", linewidth=0.7, zorder=3)
            ax.scatter(expression, row_index, s=62, color=COLORS["matched_expression"], edgecolor="white", linewidth=0.7, zorder=3)
            if "svd512_expression" in row.index:
                ax.scatter(float(row.loc["svd512_expression", "macro_f1"]), row_index, s=57, marker="D", color=COLORS["svd512_expression"], edgecolor="white", linewidth=0.7, zorder=4)
        ax.set_title(f"{chr(65 + panel)}   {DATASET_LABEL[dataset]}", loc="left", weight="bold", pad=20)
        ax.text(0.0, 1.035, subtitles[dataset], transform=ax.transAxes, fontsize=8.5, color=MID, ha="left")
        ax.set_xlim(0.84, 1.003)
        ax.set_xticks([0.85, 0.90, 0.95, 1.00])
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Pooled out-of-fold Macro F1")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([CLF_LABEL[x] for x in CLF_ORDER])
    axes[0].invert_yaxis()
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["scgpt_embeddings"], markeredgecolor="white", label=REP_LABEL["scgpt_embeddings"], markersize=7),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["matched_expression"], markeredgecolor="white", label=REP_LABEL["matched_expression"], markersize=7),
        mpl.lines.Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["svd512_expression"], markeredgecolor="white", label=REP_LABEL["svd512_expression"], markersize=6.5),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.0)
    save_figure(fig, output / "figure_2_primary_performance")


def figure_bootstrap(comparisons: pd.DataFrame, output: Path) -> None:
    data = comparisons[comparisons["comparison"] == "matched_expression_minus_scgpt"].copy()
    data["dataset"] = pd.Categorical(data["dataset"], DATASET_ORDER, ordered=True)
    data["classifier"] = pd.Categorical(data["classifier"], CLF_ORDER, ordered=True)
    data = data.sort_values(["dataset", "classifier"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.75), sharex=True, sharey=True)
    positions = np.arange(len(CLF_ORDER))
    for panel, (ax, dataset) in enumerate(zip(axes, DATASET_ORDER)):
        subset = data[data["dataset"] == dataset].set_index("classifier").reindex(CLF_ORDER)
        values = subset["difference_a_minus_b"].to_numpy(float)
        low = values - subset["group_bootstrap_ci_low"].to_numpy(float)
        high = subset["group_bootstrap_ci_high"].to_numpy(float) - values
        for index, value in enumerate(values):
            color = COLORS["matched_expression"] if value >= 0 else COLORS["scgpt_embeddings"]
            ax.errorbar(value, index, xerr=np.array([[low[index]], [high[index]]]), fmt="o", capsize=3.0, color=color, markersize=6.0, linewidth=1.5, zorder=3)
        ax.axvline(0, color=INK, linewidth=0.9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{chr(65 + panel)}   {DATASET_LABEL[dataset]}", loc="left", weight="bold", pad=10)
        ax.set_xlim(-0.076, 0.028)
        ax.set_xticks([-0.06, -0.03, 0.00, 0.02])
        ax.set_xlabel("Macro F1 difference")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([CLF_LABEL[x] for x in CLF_ORDER])
    axes[0].invert_yaxis()
    fig.text(0.5, 0.075, "← scGPT embeddings higher                         matched expression higher →", ha="center", fontsize=8.4, color=MID)
    fig.text(0.5, 0.015, "Matched expression minus scGPT embeddings; 95% paired donor-cluster bootstrap interval", ha="center", fontsize=9.2)
    fig.tight_layout(rect=(0, 0.11, 1, 0.98), w_pad=1.8)
    save_figure(fig, output / "figure_3_group_bootstrap")


def figure_calibration(temperature: pd.DataFrame, output: Path) -> None:
    keys = ["dataset", "fold", "representation", "classifier"]
    metric_specs = [
        ("ece_equal_width_10", "A   Change in ECE", "41/60 improved"),
        ("multiclass_brier", "B   Change in Brier score", "42/60 improved"),
        ("negative_log_likelihood", "C   Change in NLL", "39/60 improved"),
    ]
    before = temperature[~temperature["calibrated"].astype(bool)][keys + [m[0] for m in metric_specs]]
    after = temperature[temperature["calibrated"].astype(bool)][keys + [m[0] for m in metric_specs]]
    joined = before.merge(after, on=keys, suffixes=("_before", "_after"))
    row_keys = [(representation, classifier) for representation in REP_ORDER for classifier in ("logreg", "xgboost")]
    row_labels = [
        f"{REP_LABEL[representation]} · {'LR' if classifier == 'logreg' else 'XGB'}"
        for representation, classifier in row_keys
    ]
    dataset_style = {
        "TNBC_Breast_Cancer": ("#7057A3", "o", "TNBC"),
        "Brain_Atlas": ("#6C737A", "s", "Brain"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 4.7), sharey=True)
    for ax, (metric_name, title, count_label) in zip(axes, metric_specs):
        joined[f"delta_{metric_name}"] = joined[f"{metric_name}_after"] - joined[f"{metric_name}_before"]
        for row_index, (representation, classifier) in enumerate(row_keys):
            row = joined[(joined["representation"] == representation) & (joined["classifier"] == classifier)]
            for dataset, (color, marker, _) in dataset_style.items():
                values = row[row["dataset"] == dataset][f"delta_{metric_name}"].to_numpy(float)
                y = row_index + np.linspace(-0.10, 0.10, len(values))
                ax.scatter(values, y, s=17, color=color, marker=marker, alpha=0.42, linewidth=0, zorder=2)
                if len(values):
                    ax.scatter(np.mean(values), row_index, s=54, color=color, marker=marker, edgecolor="white", linewidth=0.7, zorder=4)
        ax.axvline(0, color=INK, linewidth=0.9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", weight="bold", pad=20)
        ax.text(0.02, 0.985, count_label, transform=ax.transAxes, fontsize=8.3, color=MID, ha="left", va="top")
        ax.set_xlabel("After minus before")
        ax.text(0.02, -0.18, "← improved", transform=ax.transAxes, fontsize=8.2, color=MID)
    axes[0].set_yticks(np.arange(len(row_keys)))
    axes[0].set_yticklabels(row_labels)
    axes[0].invert_yaxis()
    handles = [
        mpl.lines.Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, label=label, markersize=6.5)
        for color, marker, label in dataset_style.values()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.8)
    save_figure(fig, output / "figure_4_temperature_scaling")


def figure_scarcity_and_biology(scarcity: pd.DataFrame, per_class: pd.DataFrame, output: Path) -> None:
    fig = plt.figure(figsize=(10.8, 7.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15], hspace=0.48, wspace=0.46)
    scarcity_ax = fig.add_subplot(grid[0, :])
    class_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    x_order = [10, 50, 100, 250, 500]
    x_positions = np.arange(len(x_order))
    for representation in ("scgpt_embeddings", "matched_expression"):
        data = scarcity[scarcity["representation"] == representation].sort_values("labels_per_class")
        scarcity_ax.errorbar(x_positions, data["macro_f1_mean"], yerr=data["macro_f1_sd"], marker="o", capsize=3, color=COLORS[representation], label=REP_LABEL[representation], linewidth=1.8, markersize=5.5)
    scarcity_ax.set_xticks(x_positions, [str(x) for x in x_order])
    scarcity_ax.set_ylim(0.72, 0.96)
    scarcity_ax.set_xlabel("Labelled cells per class")
    scarcity_ax.set_ylabel("Donor-held-out Macro F1")
    scarcity_ax.set_title("A   Strict PBMC label scarcity", loc="left", weight="bold")
    scarcity_ax.grid(axis="y", color=GRID, lw=0.8)
    scarcity_ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 0.02))

    short_class = {
        "naive thymus-derived CD8-positive, alpha-beta T cell": "Naive CD8 T",
        "CD8-positive, alpha-beta T cell": "CD8 T",
        "CD4-positive, alpha-beta T cell": "CD4 T",
        "natural killer cell": "Natural killer",
        "naive B cell": "Naive B",
        "memory B cell": "Memory B",
        "dendritic cell, human": "Dendritic",
        "central nervous system macrophage": "CNS macrophage",
        "oligodendrocyte precursor cell": "OPC",
        "perivascular cell": "Perivascular",
        "endothelial cell": "Endothelial",
        "microglial cell": "Microglial",
        "GABAergic neuron": "GABAergic neuron",
    }
    for ax, dataset, panel in ((class_axes[0], "Indonesia_PBMC", "B"), (class_axes[1], "Brain_Atlas", "C")):
        subset = per_class[(per_class["dataset"] == dataset) & (per_class["classifier"] == "xgboost") & (per_class["representation"].isin(["scgpt_embeddings", "matched_expression"]))].copy()
        order = subset.groupby("class_name")["f1"].mean().sort_values(ascending=True).head(6).index.tolist()
        pivot = subset.pivot(index="class_name", columns="representation", values="f1").reindex(order)
        supports = subset.groupby("class_name")["support"].first().reindex(order)
        y = np.arange(len(order))
        for index in range(len(order)):
            ax.plot([pivot.iloc[index]["scgpt_embeddings"], pivot.iloc[index]["matched_expression"]], [index, index], color="#BBC0C4", lw=1.6, zorder=1)
        ax.scatter(pivot["scgpt_embeddings"], y, s=49, color=COLORS["scgpt_embeddings"], label=REP_LABEL["scgpt_embeddings"], edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(pivot["matched_expression"], y, s=49, color=COLORS["matched_expression"], label=REP_LABEL["matched_expression"], edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([short_class.get(value, value) for value in order])
        ax.set_xlim(0.72, 1.035)
        ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        ax.set_xlabel("Per-class F1 (XGBoost)")
        ax.set_title(f"{panel}   Lowest-performing {DATASET_LABEL[dataset]} classes", loc="left", weight="bold")
        ax.grid(axis="x", color=GRID, lw=0.8)
        for index, support in enumerate(supports):
            ax.text(1.007, index, f"n={int(support):,}", fontsize=7.1, color=MID, va="center", ha="left")
    fig.subplots_adjust(top=0.97, bottom=0.09, left=0.18, right=0.985)
    save_figure(fig, output / "figure_5_scarcity_and_per_class")


def supplementary_figures(results: Path, output: Path) -> None:
    confusion = pd.read_csv(results / "human_confusion_matrices_long.csv")
    confusion_label = {
        "CD14-positive monocyte": "CD14+ monocyte",
        "CD14-positive, CD16-positive monocyte": "CD14+CD16+ monocyte",
        "CD16-positive monocyte": "CD16+ monocyte",
        "CD4-positive, alpha-beta T cell": "CD4 T",
        "CD8-positive, alpha-beta T cell": "CD8 T",
        "dendritic cell, human": "Dendritic",
        "lymphocyte": "Lymphocyte",
        "memory B cell": "Memory B",
        "naive B cell": "Naive B",
        "naive thymus-derived CD8-positive, alpha-beta T cell": "Naive CD8 T",
        "natural killer cell": "NK",
        "plasma cell": "Plasma",
        "plasmacytoid dendritic cell": "pDC",
        "platelet": "Platelet",
        "GABAergic neuron": "GABAergic",
        "astrocyte": "Astrocyte",
        "central nervous system macrophage": "CNS macrophage",
        "endothelial cell": "Endothelial",
        "glutamatergic neuron": "Glutamatergic",
        "microglial cell": "Microglia",
        "oligodendrocyte": "Oligodendrocyte",
        "oligodendrocyte precursor cell": "OPC",
        "pericyte": "Pericyte",
        "perivascular cell": "Perivascular",
    }
    for dataset in ("Indonesia_PBMC", "Brain_Atlas"):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.9))
        colorbar_ax = fig.add_axes([0.925, 0.27, 0.018, 0.55])
        for panel, (ax, representation) in enumerate(zip(axes, ("scgpt_embeddings", "matched_expression"))):
            data = confusion[(confusion["dataset"] == dataset) & (confusion["representation"] == representation) & (confusion["classifier"] == "xgboost")]
            matrix = data.pivot(index="true_class", columns="predicted_class", values="row_proportion")
            matrix = matrix.rename(index=confusion_label, columns=confusion_label)
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="mako",
                vmin=0,
                vmax=1,
                square=True,
                cbar=panel == 1,
                cbar_ax=colorbar_ax if panel == 1 else None,
                cbar_kws={"label": "Row proportion"},
                linewidths=0.15,
                linecolor="#F3F4F5",
            )
            ax.set_title(f"{chr(65 + panel)}   {REP_LABEL[representation]}", loc="left", weight="bold", pad=10)
            ax.set_xlabel("Predicted portal label")
            ax.set_ylabel("Reference portal label" if panel == 0 else "")
            if panel == 1:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelrotation=53, labelsize=8.2)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
            ax.tick_params(axis="y", labelrotation=0, labelsize=8.4)
        fig.text(0.50, 0.965, f"{DATASET_LABEL[dataset]} XGBoost · row-normalised outer-fold predictions", ha="center", va="top", fontsize=10.0, color=MID)
        fig.subplots_adjust(left=0.20, right=0.90, bottom=0.31, top=0.88, wspace=0.18)
        save_figure(fig, output / f"supplementary_confusion_{dataset.lower()}")

    fold = pd.read_csv(results / "human_fold_metrics.csv")
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.5), sharex=True, sharey=True)
    for column, dataset in enumerate(DATASET_ORDER):
        for row, classifier in enumerate(CLF_ORDER):
            ax = axes[row, column]
            sub = fold[(fold["dataset"] == dataset) & (fold["classifier"] == classifier)]
            for representation in REP_ORDER:
                vals = sub[sub["representation"] == representation].sort_values("fold")
                ax.plot(vals["fold"], vals["macro_f1"], marker="o", markersize=3.5, color=COLORS[representation], linewidth=1.0)
            if row == 0:
                ax.set_title(DATASET_LABEL[dataset], weight="bold")
            if column == 0:
                ax.set_ylabel(f"{CLF_LABEL[classifier]}\nMacro F1")
            if row == 2:
                ax.set_xlabel("Outer fold")
            ax.set_xticks(range(5))
            ax.set_ylim(0, 1.02)
    handles = [mpl.lines.Line2D([0], [0], color=COLORS[r], marker="o", label=REP_LABEL[r]) for r in REP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Outer-fold variation across datasets and classifiers", weight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, output / "supplementary_outer_fold_variation")

    reliability = pd.read_csv(results / "human_reliability_bins.csv")
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 9.0), sharex=True, sharey=True)
    for row_index, dataset in enumerate(DATASET_ORDER):
        for column_index, representation in enumerate(REP_ORDER):
            ax = axes[row_index, column_index]
            sub = reliability[(reliability.dataset == dataset) & (reliability.representation == representation) & (reliability.classifier == "xgboost") & (reliability.n > 0)].copy()
            if len(sub):
                sizes = 18 + 105 * np.sqrt(sub.n / sub.n.max())
                ax.scatter(sub.mean_confidence, sub.accuracy, s=sizes, color=COLORS[representation], alpha=0.78, edgecolor="white", linewidth=0.5)
            else:
                ax.text(0.5, 0.5, "Five-fold pipeline\nnot available", ha="center", va="center", color="#666666", fontsize=10.5)
            ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if row_index == 0:
                ax.set_title(REP_LABEL[representation], weight="bold")
            if column_index == 0:
                ax.set_ylabel(f"{DATASET_LABEL[dataset]}\nObserved accuracy")
            if row_index == len(DATASET_ORDER) - 1:
                ax.set_xlabel("Mean confidence")
    fig.suptitle("XGBoost reliability by representation and atlas", weight="bold", fontsize=15)
    fig.text(0.5, 0.015, "Point area is proportional to the square root of the number of observations in the confidence bin; exact counts are supplied in CSV form.", ha="center", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    save_figure(fig, output / "supplementary_reliability_xgboost")


def main_tables(results: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    design = pd.read_csv(results / "human_dataset_design.csv")
    pooled = pd.read_csv(results / "human_pooled_metrics.csv")
    comparisons = pd.read_csv(results / "human_group_bootstrap_comparisons.csv")
    temperature = pd.read_csv(results / "temperature_scaling_fold_metrics.csv")
    scarcity = pd.read_csv(results / "strict_scarcity_summary.csv")

    lines = [r"\begin{table}[H]", r"\centering", r"\caption{Primary classifier-matched donor-held-out performance. Values are pooled out-of-fold Macro F1; every retained cell contributes exactly once.}", r"\label{tab:primary}", r"\small", r"\begin{tabular}{llrrr}", r"\toprule", r"Dataset & Classifier & scGPT embeddings & Matched expression & SVD-512 \\", r"\midrule"]
    for dataset in DATASET_ORDER:
        for classifier in CLF_ORDER:
            sub = pooled[(pooled.dataset == dataset) & (pooled.classifier == classifier)].set_index("representation")
            value = lambda representation: sub.loc[representation, "macro_f1"] if representation in sub.index else np.nan
            lines.append(f"{DATASET_LABEL[dataset]} & {CLF_LABEL[classifier]} & {metric(value('scgpt_embeddings'))} & {metric(value('matched_expression'))} & {metric(value('svd512_expression'))}" + r" \\")
        if dataset != DATASET_ORDER[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "table_primary_performance.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    primary = comparisons[comparisons.comparison == "matched_expression_minus_scgpt"].copy()
    lines = [r"\begin{table}[H]", r"\centering", r"\caption{Paired donor-cluster bootstrap contrasts for matched expression versus frozen scGPT embeddings. Intervals are conditional on the fitted out-of-fold pipelines.}", r"\label{tab:bootstrap}", r"\small", r"\begin{tabular}{llrrrr}", r"\toprule", r"Dataset & Classifier & Expression & scGPT embeddings & Difference & 95\% interval \\", r"\midrule"]
    for dataset in DATASET_ORDER:
        for classifier in CLF_ORDER:
            row = primary[(primary.dataset == dataset) & (primary.classifier == classifier)].iloc[0]
            lines.append(f"{DATASET_LABEL[dataset]} & {CLF_LABEL[classifier]} & {metric(row.macro_f1_a)} & {metric(row.macro_f1_b)} & {metric(row.difference_a_minus_b)} & [{metric(row.group_bootstrap_ci_low)}, {metric(row.group_bootstrap_ci_high)}]" + r" \\")
        if dataset != DATASET_ORDER[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "table_bootstrap.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    temp = temperature.groupby(["representation", "classifier", "calibrated"], as_index=False).agg(ece=("ece_equal_width_10", "mean"), brier=("multiclass_brier", "mean"), nll=("negative_log_likelihood", "mean"))
    lines = [r"\begin{table}[H]", r"\centering", r"\caption{Held-out-donor temperature scaling, averaged over TNBC and brain (ten outer folds). Lower values are better.}", r"\label{tab:calibration}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrrr}", r"\toprule", r"Representation & Classifier & ECE before & ECE after & Brier before & Brier after & NLL before & NLL after \\", r"\midrule"]
    for representation in REP_ORDER:
        for classifier in ("logreg", "xgboost"):
            sub = temp[(temp.representation == representation) & (temp.classifier == classifier)].set_index("calibrated")
            before, after = sub.loc[False], sub.loc[True]
            lines.append(f"{REP_LABEL[representation]} & {CLF_LABEL[classifier]} & {metric(before.ece, 4)} & {metric(after.ece, 4)} & {metric(before.brier, 4)} & {metric(after.brier, 4)} & {metric(before.nll, 4)} & {metric(after.nll, 4)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (output / "table_calibration.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pivot = scarcity.pivot(index="labels_per_class", columns="representation", values=["macro_f1_mean", "macro_f1_sd", "matched_hvgs_mean"])
    lines = [r"\begin{table}[H]", r"\centering", r"\caption{Strict Indonesia-PBMC label scarcity. Feature selection and tuning used only the labelled cells. Macro F1 is mean $\pm$ SD across five donor-held-out folds after averaging the two sampling seeds within fold.}", r"\label{tab:scarcity}", r"\small", r"\begin{tabular}{rrrr}", r"\toprule", r"Labels per class & scGPT embeddings & Matched expression & Mean matched HVGs \\", r"\midrule"]
    for labels in pivot.index:
        s_mean = pivot.loc[labels, ("macro_f1_mean", "scgpt_embeddings")]
        s_sd = pivot.loc[labels, ("macro_f1_sd", "scgpt_embeddings")]
        e_mean = pivot.loc[labels, ("macro_f1_mean", "matched_expression")]
        e_sd = pivot.loc[labels, ("macro_f1_sd", "matched_expression")]
        hvg = pivot.loc[labels, ("matched_hvgs_mean", "matched_expression")]
        lines.append(f"{int(labels)} & {metric(s_mean)} $\\pm$ {metric(s_sd)} & {metric(e_mean)} $\\pm$ {metric(e_sd)} & {hvg:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "table_scarcity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    design_lines = [r"\begin{table}[H]", r"\centering", r"\caption{Completed out-of-fold design audit.}", r"\label{tab:design-audit}", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule", r"Dataset & Donor IDs & Classes & OOF cells & Matched HVGs \\", r"\midrule"]
    for row in design.itertuples():
        hvg = f"{int(row.matched_hvgs_min)}--{int(row.matched_hvgs_max)}"
        design_lines.append(f"{DATASET_LABEL[row.dataset]} & {int(row.n_unique_groups)} & {int(row.n_classes)} & {int(row.n_oof_test_cells):,} & {hvg}" + r" \\")
    design_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "table_design_audit.tex").write_text("\n".join(design_lines) + "\n", encoding="utf-8")


def longtable(headers: list[str], rows: list[list[str]], widths: str, caption: str, label: str) -> str:
    head = " & ".join(headers) + r" \\"
    body = [r"{\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{" + widths + "}", rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule", head, r"\midrule", r"\endfirsthead", r"\multicolumn{" + str(len(headers)) + r"}{c}{\tablename\ \thetable{} continued}\\", r"\toprule", head, r"\midrule", r"\endhead", r"\midrule", r"\multicolumn{" + str(len(headers)) + r"}{r}{Continued on next page}\\", r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.append(r"\end{longtable}")
    body.append(r"}")
    return "\n".join(body)


def supplementary_body(results: Path, output: Path) -> None:
    pooled = pd.read_csv(results / "human_pooled_metrics.csv")
    fold = pd.read_csv(results / "human_fold_metrics.csv")
    comparisons = pd.read_csv(results / "human_group_bootstrap_comparisons.csv")
    params = pd.read_csv(results / "selected_hyperparameter_frequencies.csv")
    temperature = pd.read_csv(results / "temperature_scaling_summary.csv")
    per_class = pd.read_csv(results / "human_per_class_metrics.csv")
    pig = pd.read_csv(results / "pig_cross_species_metrics.csv")
    svd_audit = pd.read_csv(results / "human_svd_audit.csv")
    coverage = pd.read_csv(results / "human_pipeline_coverage.csv")

    sections = [r"\section{Completed human benchmark}", r"\input{../manuscript/generated/table_design_audit}", r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{supplementary_outer_fold_variation.pdf}\caption{Outer-fold Macro F1 for every human representation--classifier pipeline. Training folds overlap, but each outer-test donor and cell occurs in exactly one fold.}\label{fig:s-folds}\end{figure}"]

    rows = []
    for row in coverage.sort_values(["dataset", "representation", "classifier"]).itertuples():
        rows.append([DATASET_LABEL[row.dataset], REP_LABEL[row.representation], CLF_LABEL[row.classifier], "Included" if bool(row.included) else "Not available"])
    sections += [r"\clearpage\section{Five-fold pipeline coverage}", longtable(["Dataset", "Representation", "Classifier", "Status"], rows, "llll", "Pipeline coverage used by the local aggregation. A pipeline was included only when all five aligned donor-held-out prediction folds were durably available.", "tab:s-coverage")]

    rows = []
    for row in pooled.sort_values(["dataset", "classifier", "representation"]).itertuples():
        rows.append([DATASET_LABEL[row.dataset], REP_LABEL[row.representation], CLF_LABEL[row.classifier], str(int(row.n_cells)), metric(row.macro_f1), metric(row.accuracy), metric(row.balanced_accuracy), metric(row.ece_equal_width_10), metric(row.multiclass_brier), metric(row.negative_log_likelihood)])
    sections += [r"\clearpage\section{Pooled out-of-fold metrics}", longtable(["Dataset", "Representation", "Classifier", "$n$", "Macro F1", "Accuracy", "Balanced acc.", "ECE", "Brier", "NLL"], rows, "lllrrrrrrr", "Complete pooled out-of-fold metrics for the three human atlases. SVD-512 cells are blank in the main table where a five-fold classifier pipeline was not available.", "tab:s-pooled")]

    rows = []
    for row in fold.sort_values(["dataset", "classifier", "representation", "fold"]).itertuples():
        rows.append([DATASET_LABEL[row.dataset], str(int(row.fold)), REP_LABEL[row.representation], CLF_LABEL[row.classifier], str(int(row.n_test_groups)), str(int(row.n_test_cells)), metric(row.macro_f1), metric(row.accuracy), metric(row.ece_equal_width_10), metric(row.multiclass_brier)])
    sections += [r"\clearpage\section{Outer-fold metrics}", longtable(["Dataset", "Fold", "Representation", "Classifier", "Groups", "Cells", "Macro F1", "Accuracy", "ECE", "Brier"], rows, "lrllrrrrrr", "Outer-fold metrics. Fold-level values are not independent because their training partitions overlap.", "tab:s-fold")]

    rows = []
    for row in svd_audit.sort_values(["dataset", "fold"]).itertuples():
        rows.append([DATASET_LABEL[row.dataset], str(int(row.fold)), str(int(row.n_components)), f"{int(row.n_training_cells_used_to_fit):,}", metric(row.explained_variance_ratio_sum), "Yes" if bool(row.fit_restricted_to_outer_training_cells) else "No"])
    sections += [r"\clearpage\section{SVD-512 dimensionality control}", longtable(["Dataset", "Fold", "Components", "Fit cells", "Explained variance", "Training only"], rows, "lrrrrr", "Fold-specific SVD-512 audit. Decomposition parameters were fitted without outer-test cells.", "tab:s-svd")]

    rows = []
    for row in comparisons.sort_values(["comparison", "dataset", "classifier"]).itertuples():
        comparison_label = {"matched_expression_minus_scgpt": "Expression $-$ scGPT", "matched_expression_minus_svd512": "Expression $-$ SVD", "svd512_minus_scgpt": "SVD $-$ scGPT"}[row.comparison]
        rows.append([DATASET_LABEL[row.dataset], CLF_LABEL[row.classifier], comparison_label, metric(row.macro_f1_a), metric(row.macro_f1_b), metric(row.difference_a_minus_b), f"[{metric(row.group_bootstrap_ci_low)}, {metric(row.group_bootstrap_ci_high)}]", metric(row.bootstrap_probability_gt_zero)])
    sections += [r"\clearpage\section{Donor-cluster bootstrap comparisons}", longtable(["Dataset", "Classifier", "Contrast", "A", "B", "$A-B$", r"95\% interval", "$P(A>B)$"], rows, "lllrrrrr", "All paired donor-cluster bootstrap contrasts. Intervals condition on the fitted pipelines.", "tab:s-bootstrap")]

    rows = []
    for row in params.sort_values(["dataset", "representation", "classifier", "fold_frequency"], ascending=[True, True, True, False]).itertuples():
        rows.append([DATASET_LABEL[row.dataset], REP_LABEL[row.representation], CLF_LABEL[row.classifier], tex(row.selected_parameters), str(int(row.fold_frequency))])
    sections += [r"\clearpage\section{Training-only model selection}", longtable(["Dataset", "Representation", "Classifier", "Selected parameter set", "Folds"], rows, r"lll>{\raggedright\arraybackslash}p{6.8cm}r", "Frequencies of independently selected parameter sets across the five outer folds.", "tab:s-parameters")]

    rows = []
    for row in temperature.sort_values(["dataset", "representation", "classifier", "calibrated"]).itertuples():
        state = "After" if bool(row.calibrated) else "Before"
        rows.append([DATASET_LABEL[row.dataset], REP_LABEL[row.representation], CLF_LABEL[row.classifier], state, metric(row.temperature_mean), metric(row.ece_mean), metric(row.brier_mean), metric(row.nll_mean)])
    sections += [r"\clearpage\section{Temperature-scaling sensitivity}", longtable(["Dataset", "Representation", "Classifier", "State", "$T$", "ECE", "Brier", "NLL"], rows, "llllrrrr", "Calibration metrics before and after scalar temperature scaling, summarized over outer folds.", "tab:s-temperature")]

    sections += [r"\clearpage\section{Strict labelled-cell scarcity}", r"\input{../manuscript/generated/table_scarcity}", r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{supplementary_reliability_xgboost.pdf}\caption{Reliability of the final XGBoost pipelines across the human atlases. Point size encodes confidence-bin count; exact bin edges, counts, mean confidence and accuracy are supplied in \texttt{human\_reliability\_bins.csv}.}\label{fig:s-reliability}\end{figure}"]

    for dataset, figure_file in (("Indonesia_PBMC", "supplementary_confusion_indonesia_pbmc.pdf"), ("Brain_Atlas", "supplementary_confusion_brain_atlas.pdf")):
        rows = []
        subset = per_class[per_class.dataset == dataset]
        for row in subset.sort_values(["classifier", "representation", "support", "class_name"], ascending=[True, True, False, True]).itertuples():
            rows.append([tex(row.class_name), REP_LABEL[row.representation], CLF_LABEL[row.classifier], str(int(row.support)), metric(row.precision), metric(row.recall), metric(row.f1)])
        sections += [rf"\clearpage\section{{{DATASET_LABEL[dataset]} biological error profile}}", rf"\begin{{figure}}[H]\centering\includegraphics[width=\textwidth]{{{figure_file}}}\caption{{Row-normalized {DATASET_LABEL[dataset]} confusion matrices for XGBoost. Portal annotations are used as reference labels.}}\end{{figure}}", longtable(["Portal class", "Representation", "Classifier", "Support", "Precision", "Recall", "F1"], rows, r">{\raggedright\arraybackslash}p{4.2cm}llrrrr", f"Complete per-class metrics for {DATASET_LABEL[dataset]}.", f"tab:s-perclass-{DATASET_LABEL[dataset].lower()}")]

    rows = []
    for row in pig.sort_values(["classifier", "representation"]).itertuples():
        rows.append([REP_LABEL[row.representation], CLF_LABEL[row.classifier], str(int(row.n_train_cells)), str(int(row.n_test_cells)), metric(row.macro_f1), metric(row.accuracy), metric(row.ece_equal_width_10), metric(row.multiclass_brier), metric(row.negative_log_likelihood)])
    sections += [r"\clearpage\section{Pig-pancreas cross-species stress test}", r"The porcine object contained one donor ID; this analysis therefore uses a random cell split and is excluded from the human primary summary. The whole-human checkpoint was applied by exact/uppercase gene-symbol matching without an orthology map. Results should be read only as a stress test.", longtable(["Representation", "Classifier", "Train", "Test", "Macro F1", "Accuracy", "ECE", "Brier", "NLL"], rows, "llrrrrrrr", "Pig-pancreas stress-test metrics.", "tab:s-pig")]

    sections += [r"\clearpage\section{Machine-readable audit material}", r"The reproducibility archive contains the complete confusion matrices, reliability-bin counts, strict-scarcity seed-level results, selected genes for every outer fold, SVD explained-variance audits, retained hyperparameter traces, model timings, software versions, checkpoint hashes and the original-index alignment records used to verify the out-of-fold partition. The pipeline-coverage table identifies every representation--classifier combination admitted to aggregation."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    configure_style()
    figures = args.package / "figures"
    supplement_figures_dir = figures / "supplementary"
    generated = args.package / "manuscript" / "generated"
    pooled = pd.read_csv(args.results / "human_pooled_metrics.csv")
    comparisons = pd.read_csv(args.results / "human_group_bootstrap_comparisons.csv")
    temperature = pd.read_csv(args.results / "temperature_scaling_fold_metrics.csv")
    scarcity = pd.read_csv(args.results / "strict_scarcity_summary.csv")
    per_class = pd.read_csv(args.results / "human_per_class_metrics.csv")
    figure_design(figures)
    figure_performance(pooled, figures)
    figure_bootstrap(comparisons, figures)
    figure_calibration(temperature, figures)
    figure_scarcity_and_biology(scarcity, per_class, figures)
    supplementary_figures(args.results, supplement_figures_dir)
    main_tables(args.results, generated)
    supplementary_body(args.results, args.package / "supplementary" / "generated" / "supplementary_body.tex")


if __name__ == "__main__":
    main()
