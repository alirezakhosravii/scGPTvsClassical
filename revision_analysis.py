"""Reproducible analyses added for the Scientific Reports revision.

The code intentionally uses the official scGPT embedding helper from a pinned
source snapshot.  Large inputs and outputs live on the dedicated Modal volume.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path


SCGPT_SOURCE_DIR = Path("/root/scgpt_src")
SCGPT_SOURCE_COMMIT = "cebd6fa"
ATLAS_NAMES = (
    "TNBC_Breast_Cancer",
    "Indonesia_PBMC",
    "Brain_Atlas",
    "Multi_Tissue_TME",
    "Human_Pancreas",
    "Pig_Pancreas",
)


def configure_reproducibility(seed: int) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smoke_test_scgpt(data_path: str, model_dir: str, output_path: str) -> dict:
    import anndata as ad
    import numpy as np
    import torch

    if str(SCGPT_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(SCGPT_SOURCE_DIR))
    import scgpt

    configure_reproducibility(20260803)
    backed = ad.read_h5ad(data_path, backed="r")
    sample_with_raw = backed[:512, :].to_memory()
    if sample_with_raw.raw is None:
        raise ValueError("Expected integer counts in adata.raw")
    sample = sample_with_raw.raw.to_adata()
    if "feature_name" not in sample.var:
        sample.var["feature_name"] = sample.var_names.astype(str)

    embedded = scgpt.tasks.embed_data(
        sample,
        model_dir=model_dir,
        gene_col="feature_name",
        max_length=1200,
        batch_size=32,
        obs_to_save=[c for c in ["cell_type", "donor_id"] if c in sample.obs],
        device="cuda",
        use_fast_transformer=True,
        return_new_adata=True,
    )
    norms = np.linalg.norm(embedded.X, axis=1)
    payload = {
        "scgpt_version": scgpt.__version__,
        "scgpt_source_commit": SCGPT_SOURCE_COMMIT,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "input_shape": [int(sample.n_obs), int(sample.n_vars)],
        "embedding_shape": [int(x) for x in embedded.X.shape],
        "embedding_finite": bool(np.isfinite(embedded.X).all()),
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_max": float(norms.max()),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _top_label_ece(y_true, y_prob, n_bins: int = 10) -> float:
    import numpy as np

    confidence = y_prob.max(axis=1)
    correct = y_prob.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    value = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == n_bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        if mask.any():
            value += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def _adaptive_ece(y_true, y_prob, n_bins: int = 10) -> float:
    import numpy as np

    confidence = y_prob.max(axis=1)
    correct = y_prob.argmax(axis=1) == y_true
    order = np.argsort(confidence, kind="mergesort")
    value = 0.0
    for indices in np.array_split(order, n_bins):
        if len(indices):
            value += (len(indices) / len(order)) * abs(
                float(correct[indices].mean()) - float(confidence[indices].mean())
            )
    return float(value)


def _classwise_ece(y_true, y_prob, n_bins: int = 10) -> float:
    import numpy as np

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    class_values = []
    for class_index in range(y_prob.shape[1]):
        confidence = y_prob[:, class_index]
        observed = y_true == class_index
        value = 0.0
        for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            if bin_index == n_bins - 1:
                mask = (confidence >= low) & (confidence <= high)
            else:
                mask = (confidence >= low) & (confidence < high)
            if mask.any():
                value += mask.mean() * abs(
                    float(observed[mask].mean()) - float(confidence[mask].mean())
                )
        class_values.append(value)
    return float(np.mean(class_values))


def calculate_metrics(y_true, y_prob) -> dict:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
    )

    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = np.clip(y_prob, 1e-12, 1.0)
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    y_pred = y_prob.argmax(axis=1)
    one_hot = np.eye(y_prob.shape[1], dtype=np.float64)[y_true]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ece_equal_width_10": _top_label_ece(y_true, y_prob, n_bins=10),
        "ece_adaptive_10": _adaptive_ece(y_true, y_prob, n_bins=10),
        "ece_classwise_10": _classwise_ece(y_true, y_prob, n_bins=10),
        "multiclass_brier": float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))),
        "negative_log_likelihood": float(
            log_loss(y_true, y_prob, labels=np.arange(y_prob.shape[1]))
        ),
    }


def _stratified_cap(indices, y, cap: int | None, seed: int):
    import numpy as np
    from sklearn.model_selection import train_test_split

    indices = np.asarray(indices, dtype=np.int64)
    if cap is None or len(indices) <= cap:
        return indices
    selected, _ = train_test_split(
        indices,
        train_size=cap,
        random_state=seed,
        stratify=y[indices],
    )
    return np.asarray(selected, dtype=np.int64)


def _load_fold_assignments(adata, fold: int):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    labels = pd.Categorical(adata.obs["cell_type"])
    y = labels.codes.astype(np.int64)
    groups = adata.obs["donor_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260803)
    folds = list(splitter.split(np.zeros(len(y), dtype=np.uint8), y, groups))
    train_index, test_index = folds[fold]
    return y, groups, [str(x) for x in labels.categories], train_index, test_index


def _raw_slice_to_adata(adata, observation_index, variable_index=slice(None)):
    """Materialize an AnnData from Raw without Raw.to_adata obs misalignment."""
    import anndata as ad

    raw_view = adata.raw[observation_index, variable_index]
    return ad.AnnData(
        X=raw_view.X.copy(),
        obs=adata.obs.iloc[observation_index].copy(),
        var=raw_view.var.copy(),
    )


def _fit_classifier(model_key: str, X_train, y_train):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    if model_key == "logreg":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=3000,
            tol=1e-4,
            class_weight=None,
            random_state=20260803,
        )
    elif model_key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=200,
            criterion="gini",
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=20260803,
        )
    elif model_key == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            device="cuda",
            n_jobs=-1,
            random_state=20260803,
        )
    else:
        raise ValueError(model_key)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    return model


def _classifier_views(model_key: str, X_train, X_test):
    """Return numerically consistent train/test views for a classifier.

    XGBoost interprets absent entries in a scipy sparse matrix as missing
    values rather than measured expression zeros.  That behaviour is not the
    intended biological representation here and produced a severe sparse-only
    failure on validation.  Materialise both partitions for XGBoost so zeros
    have identical numeric semantics during fitting and prediction.
    """
    if model_key == "xgboost":
        import numpy as np
        import scipy.sparse as sp

        if sp.issparse(X_train):
            X_train = X_train.toarray()
        if sp.issparse(X_test):
            X_test = X_test.toarray()
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
    return X_train, X_test


def run_donor_fold(
    fold: int,
    data_path: str,
    model_dir: str,
    output_root: str,
    debug_cap_per_partition: int | None = None,
) -> dict:
    """Run one outer donor-held-out fold and persist all probabilities."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    import torch

    if str(SCGPT_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(SCGPT_SOURCE_DIR))
    import scgpt

    seed = 20260803 + fold
    configure_reproducibility(seed)
    fold_name = f"fold_{fold}" if debug_cap_per_partition is None else f"debug_fold_{fold}"
    fold_dir = Path(output_root) / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    marker = fold_dir / "complete.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))

    started = time.time()
    print(f"[{fold_name}] loading AnnData", flush=True)
    adata = ad.read_h5ad(data_path)
    if adata.raw is None:
        raise ValueError("Expected raw integer counts in adata.raw")
    y, groups, class_names, train_index, test_index = _load_fold_assignments(adata, fold)
    train_index = _stratified_cap(
        train_index, y, debug_cap_per_partition, seed=seed + 100
    )
    test_index = _stratified_cap(
        test_index, y, debug_cap_per_partition, seed=seed + 200
    )
    if set(groups[train_index]).intersection(groups[test_index]):
        raise AssertionError("Donor leakage detected between train and test")
    if set(np.unique(y[train_index])) != set(range(len(class_names))):
        raise AssertionError("Training partition is missing at least one class")
    if set(np.unique(y[test_index])) != set(range(len(class_names))):
        raise AssertionError("Test partition is missing at least one class")
    active_index = np.concatenate([train_index, test_index])
    n_train = len(train_index)
    y_active = y[active_index]

    print(f"[{fold_name}] fold-nested Seurat v3 HVG selection", flush=True)
    train_counts = _raw_slice_to_adata(adata, train_index)
    sc.pp.filter_genes(train_counts, min_counts=3)
    sc.pp.highly_variable_genes(
        train_counts,
        n_top_genes=1200,
        flavor="seurat_v3",
        subset=False,
    )
    ranked = train_counts.var.loc[train_counts.var["highly_variable"]].copy()
    ranked = ranked.sort_values("highly_variable_rank", kind="mergesort")
    if "feature_name" not in ranked:
        ranked["feature_name"] = ranked.index.astype(str)

    vocab_dict = json.loads((Path(model_dir) / "vocab.json").read_text(encoding="utf-8"))
    matched = ranked[ranked["feature_name"].astype(str).isin(vocab_dict)].copy()
    selected_var_names = matched.index.astype(str).tolist()
    selected_feature_names = matched["feature_name"].astype(str).tolist()
    if len(selected_var_names) < 100:
        raise ValueError(f"Implausibly low vocabulary match: {len(selected_var_names)} genes")
    del train_counts, ranked, matched

    gene_payload = {
        "fold": fold,
        "n_hvg_requested": 1200,
        "n_hvg_matched": len(selected_var_names),
        "var_names": selected_var_names,
        "feature_names": selected_feature_names,
    }
    (fold_dir / "selected_genes.json").write_text(
        json.dumps(gene_payload, indent=2), encoding="utf-8"
    )

    print(f"[{fold_name}] official scGPT embedding of {len(active_index):,} cells", flush=True)
    raw_selected = _raw_slice_to_adata(adata, active_index, selected_var_names)
    raw_selected.var["feature_name"] = selected_feature_names
    if sp.issparse(raw_selected.X):
        # Some CELLxGENE count matrices retain explicitly stored zeros, so
        # getnnz() is not a valid test for an actually expressed input gene.
        nonzero_input = np.asarray(raw_selected.X.sum(axis=1)).ravel() > 0
    else:
        nonzero_input = np.asarray(raw_selected.X).sum(axis=1) > 0
    n_zero_train = int((~nonzero_input[:n_train]).sum())
    n_zero_test = int((~nonzero_input[n_train:]).sum())
    if n_zero_train or n_zero_test:
        print(
            f"[{fold_name}] excluding {n_zero_train} training and {n_zero_test} test cells "
            "with zero counts across the shared input genes",
            flush=True,
        )
        raw_selected = raw_selected[nonzero_input].copy()
        active_index = active_index[nonzero_input]
        y_active = y_active[nonzero_input]
        n_train = int(nonzero_input[:n_train].sum())
        train_index = active_index[:n_train]
        test_index = active_index[n_train:]
        if len(np.unique(y_active[:n_train])) != len(class_names):
            raise AssertionError("Shared-input QC removed a training class")
        if len(np.unique(y_active[n_train:])) != len(class_names):
            raise AssertionError("Shared-input QC removed a test class")
    embedded = scgpt.tasks.embed_data(
        raw_selected,
        model_dir=model_dir,
        gene_col="feature_name",
        max_length=1200,
        batch_size=256,
        obs_to_save=None,
        device="cuda",
        use_fast_transformer=True,
        return_new_adata=True,
    )
    embeddings = np.asarray(embedded.X, dtype=np.float32)
    if embeddings.shape != (len(active_index), 512) or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid embeddings: shape={embeddings.shape}")
    np.save(fold_dir / "active_indices.npy", active_index)
    np.save(fold_dir / "embeddings.npy", embeddings)
    del raw_selected, embedded
    torch.cuda.empty_cache()

    print(f"[{fold_name}] loading matched log-normalized expression features", flush=True)
    expression = adata[active_index, selected_var_names].X
    expression = expression.tocsr().astype(np.float32) if sp.issparse(expression) else np.asarray(expression, dtype=np.float32)
    X_by_representation = {
        "scgpt_embeddings": embeddings,
        "matched_hvgs": expression,
    }
    del adata

    pipeline_specs = [
        ("scGPT embeddings plus Logistic Regression", "scgpt_embeddings", "logreg"),
        ("scGPT embeddings plus Random Forest", "scgpt_embeddings", "random_forest"),
        ("scGPT embeddings plus XGBoost", "scgpt_embeddings", "xgboost"),
        ("Matched HVGs plus Logistic Regression", "matched_hvgs", "logreg"),
        ("Matched HVGs plus Random Forest", "matched_hvgs", "random_forest"),
        ("Matched HVGs plus XGBoost", "matched_hvgs", "xgboost"),
    ]
    result_rows = []
    for model_name, representation, classifier in pipeline_specs:
        print(f"[{fold_name}] fitting {model_name}", flush=True)
        X = X_by_representation[representation]
        fit_started = time.time()
        X_train_view, X_test_view = _classifier_views(
            classifier, X[:n_train], X[n_train:]
        )
        fitted = _fit_classifier(classifier, X_train_view, y_active[:n_train])
        y_probability = fitted.predict_proba(X_test_view).astype(np.float32)
        if not np.array_equal(fitted.classes_, np.arange(len(class_names))):
            raise AssertionError(f"Unexpected class order for {model_name}: {fitted.classes_}")
        metrics = calculate_metrics(y_active[n_train:], y_probability)
        elapsed = time.time() - fit_started
        slug = (
            model_name.lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("-", "_")
        )
        np.savez_compressed(
            fold_dir / f"predictions__{slug}.npz",
            fold=np.int64(fold),
            test_index=test_index.astype(np.int64),
            donor_id=groups[test_index].astype(str),
            y_true=y_active[n_train:].astype(np.int16),
            y_prob=y_probability,
            class_names=np.asarray(class_names, dtype=str),
            model_name=np.asarray(model_name),
        )
        result_rows.append(
            {
                "fold": fold,
                "model": model_name,
                "representation": representation,
                "classifier": classifier,
                "n_train_cells": n_train,
                "n_test_cells": len(test_index),
                "n_train_donors": len(np.unique(groups[train_index])),
                "n_test_donors": len(np.unique(groups[test_index])),
                "n_matched_hvgs": len(selected_var_names),
                "fit_and_predict_seconds": elapsed,
                **metrics,
            }
        )
        del fitted, y_probability, X_train_view, X_test_view
        if classifier == "xgboost":
            torch.cuda.empty_cache()

    results = pd.DataFrame(result_rows)
    results.to_csv(fold_dir / "metrics.csv", index=False)
    fold_payload = {
        "fold": fold,
        "debug_cap_per_partition": debug_cap_per_partition,
        "n_train_cells": n_train,
        "n_test_cells": len(test_index),
        "n_train_donors": len(np.unique(groups[train_index])),
        "n_test_donors": len(np.unique(groups[test_index])),
        "n_classes": len(class_names),
        "n_matched_hvgs": len(selected_var_names),
        "scgpt_version": scgpt.__version__,
        "scgpt_source_commit": SCGPT_SOURCE_COMMIT,
        "total_seconds": time.time() - started,
        "metrics": result_rows,
    }
    marker.write_text(json.dumps(fold_payload, indent=2), encoding="utf-8")
    print(f"[{fold_name}] complete in {fold_payload['total_seconds'] / 60:.1f} minutes", flush=True)
    return fold_payload


def run_atlas_cell_split(
    dataset_name: str,
    data_path: str,
    model_dir: str,
    output_root: str,
) -> dict:
    """Run the corrected within-atlas matched benchmark on one atlas."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    import torch
    from sklearn.model_selection import train_test_split

    if dataset_name not in ATLAS_NAMES:
        raise KeyError(dataset_name)
    if str(SCGPT_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(SCGPT_SOURCE_DIR))
    import scgpt

    seed = 20261003 + ATLAS_NAMES.index(dataset_name)
    configure_reproducibility(seed)
    destination = Path(output_root) / dataset_name
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "complete.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))

    started = time.time()
    print(f"[{dataset_name}] loading AnnData", flush=True)
    adata = ad.read_h5ad(data_path)
    if adata.raw is None:
        raise ValueError(f"{dataset_name} lacks raw integer counts")
    label_series = adata.obs["cell_type"].astype("string")
    counts = label_series.value_counts(dropna=False)
    retained_labels = counts[counts > 5].index
    retained_index = np.flatnonzero(label_series.isin(retained_labels).to_numpy())
    retained_categorical = pd.Categorical(label_series.iloc[retained_index])
    y = retained_categorical.codes.astype(np.int64)
    class_names = [str(value) for value in retained_categorical.categories]
    train_position, test_position = train_test_split(
        np.arange(len(retained_index), dtype=np.int64),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    train_index = retained_index[train_position]
    test_index = retained_index[test_position]
    active_index = np.concatenate([train_index, test_index])
    n_train = len(train_index)
    y_active = np.concatenate([y[train_position], y[test_position]])

    print(f"[{dataset_name}] training-only Seurat v3 HVG selection", flush=True)
    train_counts = _raw_slice_to_adata(adata, train_index)
    sc.pp.filter_genes(train_counts, min_counts=3)
    sc.pp.highly_variable_genes(
        train_counts,
        n_top_genes=1200,
        flavor="seurat_v3",
        subset=False,
    )
    ranked = train_counts.var.loc[train_counts.var["highly_variable"]].copy()
    ranked = ranked.sort_values("highly_variable_rank", kind="mergesort")
    if "feature_name" not in ranked:
        ranked["feature_name"] = ranked.index.astype(str)
    vocab_dict = json.loads((Path(model_dir) / "vocab.json").read_text(encoding="utf-8"))

    def map_feature(value):
        value = str(value)
        if value in vocab_dict:
            return value
        upper = value.upper()
        return upper if upper in vocab_dict else None

    ranked["mapped_feature_name"] = ranked["feature_name"].map(map_feature)
    matched = ranked[ranked["mapped_feature_name"].notna()].copy()
    matched = matched.drop_duplicates("mapped_feature_name", keep="first")
    selected_var_names = matched.index.astype(str).tolist()
    selected_feature_names = matched["mapped_feature_name"].astype(str).tolist()
    if len(selected_var_names) < 100:
        raise ValueError(
            f"{dataset_name} has implausibly low vocabulary match: {len(selected_var_names)} genes"
        )
    del train_counts, ranked, matched
    (destination / "selected_genes.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "n_hvg_requested": 1200,
                "n_hvg_matched": len(selected_var_names),
                "var_names": selected_var_names,
                "feature_names": selected_feature_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[{dataset_name}] official scGPT embedding of {len(active_index):,} cells "
        f"with {len(selected_var_names):,} matched HVGs",
        flush=True,
    )
    raw_selected = _raw_slice_to_adata(adata, active_index, selected_var_names)
    raw_selected.var["feature_name"] = selected_feature_names
    if sp.issparse(raw_selected.X):
        # CELLxGENE matrices can retain explicitly stored zeros. Remove them
        # before testing rows because scGPT requires at least one expressed
        # gene per cell in the selected input space.
        cleaned_counts = raw_selected.X.tocsr(copy=True)
        cleaned_counts.eliminate_zeros()
        raw_selected.X = cleaned_counts
        nonzero_input = np.diff(cleaned_counts.indptr) > 0
    else:
        nonzero_input = np.count_nonzero(np.asarray(raw_selected.X), axis=1) > 0
    n_zero_train = int((~nonzero_input[:n_train]).sum())
    n_zero_test = int((~nonzero_input[n_train:]).sum())
    if n_zero_train or n_zero_test:
        print(
            f"[{dataset_name}] excluding {n_zero_train} training and {n_zero_test} test cells "
            "with zero counts across the shared input genes",
            flush=True,
        )
        raw_selected = raw_selected[nonzero_input].copy()
        active_index = active_index[nonzero_input]
        y_active = y_active[nonzero_input]
        n_train = int(nonzero_input[:n_train].sum())
        train_index = active_index[:n_train]
        test_index = active_index[n_train:]
        if len(np.unique(y_active[:n_train])) != len(class_names):
            raise AssertionError("Shared-input QC removed a training class")
        if len(np.unique(y_active[n_train:])) != len(class_names):
            raise AssertionError("Shared-input QC removed a test class")
    embedded = scgpt.tasks.embed_data(
        raw_selected,
        model_dir=model_dir,
        gene_col="feature_name",
        max_length=1200,
        batch_size=256,
        obs_to_save=None,
        device="cuda",
        use_fast_transformer=True,
        return_new_adata=True,
    )
    embeddings = np.asarray(embedded.X, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    if embeddings.shape != (len(active_index), 512) or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid embeddings for {dataset_name}: {embeddings.shape}")
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(f"Embeddings are not unit normalized for {dataset_name}")
    np.save(destination / "active_indices.npy", active_index)
    np.save(destination / "embeddings.npy", embeddings)
    del embedded
    torch.cuda.empty_cache()

    print(f"[{dataset_name}] normalizing matched raw counts for expression baselines", flush=True)
    sc.pp.normalize_total(raw_selected, target_sum=1e4)
    sc.pp.log1p(raw_selected)
    expression = raw_selected.X
    expression = (
        expression.tocsr().astype(np.float32)
        if sp.issparse(expression)
        else np.asarray(expression, dtype=np.float32)
    )
    del raw_selected, adata
    representations = {"scgpt_embeddings": embeddings, "matched_hvgs": expression}
    specs = [
        ("scGPT embeddings plus Logistic Regression", "scgpt_embeddings", "logreg"),
        ("scGPT embeddings plus Random Forest", "scgpt_embeddings", "random_forest"),
        ("scGPT embeddings plus XGBoost", "scgpt_embeddings", "xgboost"),
        ("Matched HVGs plus Logistic Regression", "matched_hvgs", "logreg"),
        ("Matched HVGs plus Random Forest", "matched_hvgs", "random_forest"),
        ("Matched HVGs plus XGBoost", "matched_hvgs", "xgboost"),
    ]
    result_rows = []
    for model_name, representation, classifier in specs:
        print(f"[{dataset_name}] fitting {model_name}", flush=True)
        fit_started = time.time()
        X = representations[representation]
        X_train_view, X_test_view = _classifier_views(
            classifier, X[:n_train], X[n_train:]
        )
        fitted = _fit_classifier(classifier, X_train_view, y_active[:n_train])
        probability = fitted.predict_proba(X_test_view).astype(np.float32)
        if not np.array_equal(fitted.classes_, np.arange(len(class_names))):
            raise AssertionError(f"Unexpected class order for {dataset_name} {model_name}")
        metrics = calculate_metrics(y_active[n_train:], probability)
        np.savez_compressed(
            destination / f"predictions__{_model_slug(model_name)}.npz",
            dataset=np.asarray(dataset_name),
            test_index=test_index.astype(np.int64),
            y_true=y_active[n_train:].astype(np.int16),
            y_prob=probability,
            class_names=np.asarray(class_names, dtype=str),
            model_name=np.asarray(model_name),
        )
        result_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "representation": representation,
                "classifier": classifier,
                "n_cells": len(active_index),
                "n_cells_before_shared_input_qc": len(retained_index),
                "n_zero_input_excluded": n_zero_train + n_zero_test,
                "n_train_cells": n_train,
                "n_test_cells": len(test_index),
                "n_classes": len(class_names),
                "n_matched_hvgs": len(selected_var_names),
                "fit_and_predict_seconds": time.time() - fit_started,
                **metrics,
            }
        )
        del fitted, probability, X_train_view, X_test_view
        if classifier == "xgboost":
            torch.cuda.empty_cache()
    pd.DataFrame(result_rows).to_csv(destination / "metrics.csv", index=False)
    payload = {
        "dataset": dataset_name,
        "n_cells": len(active_index),
        "n_cells_before_shared_input_qc": len(retained_index),
        "n_zero_input_train_excluded": n_zero_train,
        "n_zero_input_test_excluded": n_zero_test,
        "n_train_cells": n_train,
        "n_test_cells": len(test_index),
        "n_classes": len(class_names),
        "class_names": class_names,
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "n_matched_hvgs": len(selected_var_names),
        "scgpt_version": scgpt.__version__,
        "scgpt_source_commit": SCGPT_SOURCE_COMMIT,
        "total_seconds": time.time() - started,
        "metrics": result_rows,
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[{dataset_name}] complete in {payload['total_seconds'] / 60:.1f} minutes", flush=True)
    return payload


def rerun_donor_fold_with_raw_normalized_expression(
    fold: int,
    data_path: str,
    cache_root: str,
    output_root: str,
) -> dict:
    """Refit donor-fold pipelines with classical features derived from raw counts."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    import torch

    configure_reproducibility(20261103 + fold)
    source = Path(cache_root) / f"fold_{fold}"
    destination = Path(output_root) / f"fold_{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "complete.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    started = time.time()
    source_complete = json.loads((source / "complete.json").read_text(encoding="utf-8"))
    genes = json.loads((source / "selected_genes.json").read_text(encoding="utf-8"))
    active_index = np.load(source / "active_indices.npy")
    embeddings = np.load(source / "embeddings.npy", mmap_mode="r")
    n_train = int(source_complete["n_train_cells"])

    print(f"[normalized fold_{fold}] loading raw counts", flush=True)
    adata = ad.read_h5ad(data_path)
    y_all, groups, class_names, train_index, test_index = _load_fold_assignments(adata, fold)
    if not np.array_equal(active_index[:n_train], train_index):
        raise AssertionError("Cached donor-fold training indices differ")
    if not np.array_equal(active_index[n_train:], test_index):
        raise AssertionError("Cached donor-fold test indices differ")
    y_active = y_all[active_index]
    expression_adata = _raw_slice_to_adata(adata, active_index, genes["var_names"])
    sc.pp.normalize_total(expression_adata, target_sum=1e4)
    sc.pp.log1p(expression_adata)
    expression = expression_adata.X
    expression = (
        expression.tocsr().astype(np.float32)
        if sp.issparse(expression)
        else np.asarray(expression, dtype=np.float32)
    )
    del expression_adata, adata

    representations = {"scgpt_embeddings": embeddings, "matched_hvgs": expression}
    specs = [
        ("scGPT embeddings plus Logistic Regression", "scgpt_embeddings", "logreg"),
        ("scGPT embeddings plus Random Forest", "scgpt_embeddings", "random_forest"),
        ("scGPT embeddings plus XGBoost", "scgpt_embeddings", "xgboost"),
        ("Matched HVGs plus Logistic Regression", "matched_hvgs", "logreg"),
        ("Matched HVGs plus Random Forest", "matched_hvgs", "random_forest"),
        ("Matched HVGs plus XGBoost", "matched_hvgs", "xgboost"),
    ]
    result_rows = []
    for model_name, representation, classifier in specs:
        print(f"[normalized fold_{fold}] fitting {model_name}", flush=True)
        fit_started = time.time()
        X = representations[representation]
        X_train_view, X_test_view = _classifier_views(
            classifier, X[:n_train], X[n_train:]
        )
        fitted = _fit_classifier(classifier, X_train_view, y_active[:n_train])
        probability = fitted.predict_proba(X_test_view).astype(np.float32)
        if not np.array_equal(fitted.classes_, np.arange(len(class_names))):
            raise AssertionError(f"Unexpected class order for {model_name}")
        metrics = calculate_metrics(y_active[n_train:], probability)
        np.savez_compressed(
            destination / f"predictions__{_model_slug(model_name)}.npz",
            fold=np.int64(fold),
            test_index=test_index.astype(np.int64),
            donor_id=groups[test_index].astype(str),
            y_true=y_active[n_train:].astype(np.int16),
            y_prob=probability,
            class_names=np.asarray(class_names, dtype=str),
            model_name=np.asarray(model_name),
        )
        result_rows.append(
            {
                "fold": fold,
                "model": model_name,
                "representation": representation,
                "classifier": classifier,
                "n_train_cells": n_train,
                "n_test_cells": len(test_index),
                "n_train_donors": len(np.unique(groups[train_index])),
                "n_test_donors": len(np.unique(groups[test_index])),
                "n_matched_hvgs": int(genes["n_hvg_matched"]),
                "expression_preprocessing": "raw counts normalized to 10000 then log1p",
                "fit_and_predict_seconds": time.time() - fit_started,
                **metrics,
            }
        )
        del fitted, probability, X_train_view, X_test_view
        if classifier == "xgboost":
            torch.cuda.empty_cache()
    pd.DataFrame(result_rows).to_csv(destination / "metrics.csv", index=False)
    payload = {
        "fold": fold,
        "n_train_cells": n_train,
        "n_test_cells": len(test_index),
        "n_train_donors": len(np.unique(groups[train_index])),
        "n_test_donors": len(np.unique(groups[test_index])),
        "n_classes": len(class_names),
        "n_matched_hvgs": int(genes["n_hvg_matched"]),
        "expression_preprocessing": "raw counts normalized to 10000 then log1p",
        "total_seconds": time.time() - started,
        "metrics": result_rows,
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _model_slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")


def _softmax(logits):
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _fit_temperature(logits, y_true) -> float:
    """Fit one positive temperature by held-out multiclass NLL."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        scaled = logits / temperature
        return float(
            np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y_true)), y_true])
        )

    result = minimize_scalar(
        objective,
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def run_donor_secondary_fold(
    fold: int,
    data_path: str,
    fold_root: str,
    output_root: str,
    labels_per_class=(10, 25, 50, 100, 250, 500),
    scarcity_seeds=(20260811, 20260812, 20260813, 20260814, 20260815),
) -> dict:
    """Run label-scarcity and held-out-donor temperature-scaling analyses."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import torch
    from sklearn.model_selection import StratifiedGroupKFold

    configure_reproducibility(20260803 + fold)
    source = Path(fold_root) / f"fold_{fold}"
    destination = Path(output_root) / f"fold_{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "complete.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))

    complete = json.loads((source / "complete.json").read_text(encoding="utf-8"))
    genes = json.loads((source / "selected_genes.json").read_text(encoding="utf-8"))
    active_index = np.load(source / "active_indices.npy")
    embeddings = np.load(source / "embeddings.npy", mmap_mode="r")
    n_train = int(complete["n_train_cells"])
    if len(active_index) != embeddings.shape[0] or n_train >= len(active_index):
        raise AssertionError("Cached fold arrays are inconsistent")

    print(f"[secondary fold_{fold}] loading labels and matched expression", flush=True)
    adata = ad.read_h5ad(data_path)
    y_all, groups_all, class_names, train_index, test_index = _load_fold_assignments(adata, fold)
    if not np.array_equal(active_index[:n_train], train_index):
        raise AssertionError("Cached training indices differ from reconstructed donor fold")
    if not np.array_equal(active_index[n_train:], test_index):
        raise AssertionError("Cached test indices differ from reconstructed donor fold")
    y_active = y_all[active_index]
    groups_active = groups_all[active_index]
    expression_adata = _raw_slice_to_adata(adata, active_index, genes["var_names"])
    import scanpy as sc

    sc.pp.normalize_total(expression_adata, target_sum=1e4)
    sc.pp.log1p(expression_adata)
    expression = expression_adata.X
    expression = (
        expression.tocsr().astype(np.float32)
        if sp.issparse(expression)
        else np.asarray(expression, dtype=np.float32)
    )
    del expression_adata, adata

    representations = {
        "scGPT embeddings": embeddings,
        "Matched HVGs": expression,
    }
    y_train = y_active[:n_train]
    y_test = y_active[n_train:]
    n_classes = len(class_names)
    scarcity_rows = []
    for labels in labels_per_class:
        for scarcity_seed in scarcity_seeds:
            rng = np.random.default_rng(int(scarcity_seed + 1000 * fold + labels))
            selected_parts = []
            for class_index in range(n_classes):
                candidates = np.flatnonzero(y_train == class_index)
                if len(candidates) < labels:
                    raise ValueError(
                        f"Fold {fold} class {class_names[class_index]} has only "
                        f"{len(candidates)} training cells for a target of {labels}"
                    )
                selected_parts.append(rng.choice(candidates, size=labels, replace=False))
            selected = np.concatenate(selected_parts)
            rng.shuffle(selected)
            for representation_name, X in representations.items():
                for classifier in ("logreg", "xgboost"):
                    model_name = f"{representation_name} plus " + (
                        "Logistic Regression" if classifier == "logreg" else "XGBoost"
                    )
                    print(
                        f"[secondary fold_{fold}] {labels}/class seed {scarcity_seed}: {model_name}",
                        flush=True,
                    )
                    started = time.time()
                    X_train_view, X_test_view = _classifier_views(
                        classifier, X[selected], X[n_train:]
                    )
                    fitted = _fit_classifier(classifier, X_train_view, y_train[selected])
                    probability = fitted.predict_proba(X_test_view).astype(np.float32)
                    if not np.array_equal(fitted.classes_, np.arange(n_classes)):
                        raise AssertionError(f"Unexpected class order for {model_name}")
                    scarcity_rows.append(
                        {
                            "fold": fold,
                            "labels_per_class": int(labels),
                            "scarcity_seed": int(scarcity_seed),
                            "total_training_labels": int(len(selected)),
                            "representation": representation_name,
                            "classifier": classifier,
                            "model": model_name,
                            "fit_and_predict_seconds": time.time() - started,
                            **calculate_metrics(y_test, probability),
                        }
                    )
                    del fitted, probability, X_train_view, X_test_view
                    if classifier == "xgboost":
                        torch.cuda.empty_cache()
    pd.DataFrame(scarcity_rows).to_csv(destination / "label_scarcity_metrics.csv", index=False)

    print(f"[secondary fold_{fold}] constructing inner donor calibration split", flush=True)
    inner = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=20260903 + fold,
    )
    inner_candidates = list(
        inner.split(
            np.zeros(n_train, dtype=np.uint8),
            y_train,
            groups_active[:n_train],
        )
    )
    fit_position = calibration_position = None
    for candidate_fit, candidate_calibration in inner_candidates:
        if (
            len(np.unique(y_train[candidate_fit])) == n_classes
            and len(np.unique(y_train[candidate_calibration])) == n_classes
        ):
            fit_position = candidate_fit
            calibration_position = candidate_calibration
            break
    if fit_position is None or calibration_position is None:
        raise RuntimeError("Could not construct an inner donor split containing every class")
    if set(groups_active[fit_position]).intersection(groups_active[calibration_position]):
        raise AssertionError("Donor leakage between classifier-fit and calibration partitions")

    temperature_rows = []
    temperature_probability_dir = destination / "temperature_probabilities"
    temperature_probability_dir.mkdir(exist_ok=True)
    for representation_name, X in representations.items():
        print(
            f"[secondary fold_{fold}] temperature scaling {representation_name} plus Logistic Regression",
            flush=True,
        )
        fitted = _fit_classifier("logreg", X[fit_position], y_train[fit_position])
        if not np.array_equal(fitted.classes_, np.arange(n_classes)):
            raise AssertionError("Unexpected class order in temperature-scaling classifier")
        calibration_logits = fitted.decision_function(X[calibration_position])
        temperature = _fit_temperature(
            calibration_logits,
            y_train[calibration_position],
        )
        test_logits = fitted.decision_function(X[n_train:])
        uncalibrated = _softmax(test_logits).astype(np.float32)
        calibrated = _softmax(test_logits / temperature).astype(np.float32)
        if not np.array_equal(uncalibrated.argmax(axis=1), calibrated.argmax(axis=1)):
            raise AssertionError("Positive temperature scaling unexpectedly changed class predictions")
        pipeline = f"{representation_name} plus Logistic Regression"
        for calibrated_flag, probability in ((False, uncalibrated), (True, calibrated)):
            temperature_rows.append(
                {
                    "fold": fold,
                    "pipeline": pipeline,
                    "calibrated": calibrated_flag,
                    "temperature": temperature,
                    "n_classifier_fit_cells": int(len(fit_position)),
                    "n_calibration_cells": int(len(calibration_position)),
                    "n_classifier_fit_donors": int(len(np.unique(groups_active[fit_position]))),
                    "n_calibration_donors": int(len(np.unique(groups_active[calibration_position]))),
                    "n_test_cells": int(len(y_test)),
                    "n_test_donors": int(len(np.unique(groups_active[n_train:]))),
                    **calculate_metrics(y_test, probability),
                }
            )
        slug = _model_slug(representation_name)
        np.savez_compressed(
            temperature_probability_dir / f"{slug}.npz",
            fold=np.int64(fold),
            test_index=test_index.astype(np.int64),
            donor_id=groups_active[n_train:].astype(str),
            y_true=y_test.astype(np.int16),
            uncalibrated_probability=uncalibrated,
            calibrated_probability=calibrated,
            temperature=np.float64(temperature),
            class_names=np.asarray(class_names, dtype=str),
        )
        del fitted, calibration_logits, test_logits, uncalibrated, calibrated
    pd.DataFrame(temperature_rows).to_csv(destination / "temperature_scaling_metrics.csv", index=False)

    payload = {
        "fold": fold,
        "labels_per_class": [int(value) for value in labels_per_class],
        "scarcity_seeds": [int(value) for value in scarcity_seeds],
        "n_label_scarcity_fits": len(scarcity_rows),
        "n_temperature_rows": len(temperature_rows),
        "n_classifier_fit_donors": int(len(np.unique(groups_active[fit_position]))),
        "n_calibration_donors": int(len(np.unique(groups_active[calibration_position]))),
        "n_test_donors": int(len(np.unique(groups_active[n_train:]))),
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _macro_f1_from_confusions(confusions):
    import numpy as np

    confusions = np.asarray(confusions, dtype=np.float64)
    diagonal = np.diagonal(confusions, axis1=-2, axis2=-1)
    row_total = confusions.sum(axis=-1)
    column_total = confusions.sum(axis=-2)
    denominator = row_total + column_total
    per_class = np.divide(
        2.0 * diagonal,
        denominator,
        out=np.zeros_like(diagonal),
        where=denominator > 0,
    )
    return per_class.mean(axis=-1)


def aggregate_donor_cv(
    fold_root: str,
    output_root: str,
    expected_n_cells: int = 462034,
    n_bootstrap: int = 5000,
) -> dict:
    """Aggregate OOF predictions and compute efficient donor/cell bootstraps."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    root = Path(fold_root)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    fold_metrics = pd.concat(
        [pd.read_csv(root / f"fold_{fold}" / "metrics.csv") for fold in range(5)],
        ignore_index=True,
    )
    model_names = fold_metrics["model"].drop_duplicates().tolist()
    fold_metrics.to_csv(destination / "donor_cv_fold_metrics.csv", index=False)

    oof = {}
    for model_name in model_names:
        parts = []
        for fold in range(5):
            path = root / f"fold_{fold}" / f"predictions__{_model_slug(model_name)}.npz"
            data = np.load(path, allow_pickle=False)
            parts.append(
                {
                    "test_index": data["test_index"],
                    "donor_id": data["donor_id"].astype(str),
                    "y_true": data["y_true"].astype(np.int64),
                    "y_prob": data["y_prob"].astype(np.float32),
                    "class_names": data["class_names"].astype(str),
                }
            )
        indices = np.concatenate([part["test_index"] for part in parts])
        if len(indices) != expected_n_cells or len(np.unique(indices)) != expected_n_cells:
            raise AssertionError(f"OOF coverage failed for {model_name}: {len(indices)} rows")
        order = np.argsort(indices, kind="mergesort")
        if not np.array_equal(indices[order], np.arange(expected_n_cells)):
            raise AssertionError(f"OOF indices are not an exact partition for {model_name}")
        y_true = np.concatenate([part["y_true"] for part in parts])[order]
        donor_id = np.concatenate([part["donor_id"] for part in parts])[order]
        y_prob = np.concatenate([part["y_prob"] for part in parts], axis=0)[order]
        class_names = parts[0]["class_names"]
        oof[model_name] = {
            "y_true": y_true,
            "donor_id": donor_id,
            "y_prob": y_prob,
            "class_names": class_names,
        }
        np.savez_compressed(
            destination / f"oof_predictions__{_model_slug(model_name)}.npz",
            y_true=y_true.astype(np.int16),
            donor_id=donor_id.astype(str),
            y_prob=y_prob,
            class_names=class_names.astype(str),
            model_name=np.asarray(model_name),
        )

    reference_y = next(iter(oof.values()))["y_true"]
    reference_donor = next(iter(oof.values()))["donor_id"]
    for values in oof.values():
        if not np.array_equal(values["y_true"], reference_y):
            raise AssertionError("OOF truth vectors differ between models")
        if not np.array_equal(values["donor_id"], reference_donor):
            raise AssertionError("OOF donor vectors differ between models")

    donors = np.unique(reference_donor)
    donor_lookup = {donor: index for index, donor in enumerate(donors)}
    donor_code = np.asarray([donor_lookup[d] for d in reference_donor], dtype=np.int16)
    n_donors = len(donors)
    n_classes = next(iter(oof.values()))["y_prob"].shape[1]
    n_bins = 10
    rng = np.random.default_rng(20260803)
    donor_weights = rng.multinomial(
        n_donors, np.repeat(1.0 / n_donors, n_donors), size=n_bootstrap
    ).astype(np.float64)

    pooled_rows = []
    interval_rows = []
    donor_boot_f1 = {}
    reliability_records = []
    for model_name, values in oof.items():
        y_true = values["y_true"]
        y_prob = values["y_prob"].astype(np.float64)
        y_pred = y_prob.argmax(axis=1)
        pooled = calculate_metrics(y_true, y_prob)
        pooled_rows.append({"model": model_name, "n_cells": len(y_true), "n_donors": n_donors, **pooled})

        donor_confusion = np.zeros((n_donors, n_classes, n_classes), dtype=np.int64)
        np.add.at(donor_confusion, (donor_code, y_true, y_pred), 1)
        donor_confusion_flat = donor_confusion.reshape(n_donors, -1)
        bootstrap_confusion = (donor_weights @ donor_confusion_flat).reshape(
            n_bootstrap, n_classes, n_classes
        )
        boot_f1 = _macro_f1_from_confusions(bootstrap_confusion)
        donor_boot_f1[model_name] = boot_f1

        overall_confusion = donor_confusion.sum(axis=0)
        cell_probabilities = overall_confusion.ravel() / overall_confusion.sum()
        cell_confusions = rng.multinomial(
            int(overall_confusion.sum()), cell_probabilities, size=n_bootstrap
        ).reshape(n_bootstrap, n_classes, n_classes)
        cell_boot_f1 = _macro_f1_from_confusions(cell_confusions)

        one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
        brier_cell = np.sum((y_prob - one_hot) ** 2, axis=1)
        nll_cell = -np.log(np.clip(y_prob[np.arange(len(y_true)), y_true], 1e-12, 1.0))
        donor_n = np.bincount(donor_code, minlength=n_donors).astype(np.float64)
        donor_brier = np.bincount(donor_code, weights=brier_cell, minlength=n_donors)
        donor_nll = np.bincount(donor_code, weights=nll_cell, minlength=n_donors)
        bootstrap_n = donor_weights @ donor_n
        boot_brier = (donor_weights @ donor_brier) / bootstrap_n
        boot_nll = (donor_weights @ donor_nll) / bootstrap_n

        confidence = y_prob.max(axis=1)
        correct = (y_pred == y_true).astype(np.float64)
        bin_index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
        donor_bin_count = np.zeros((n_donors, n_bins), dtype=np.float64)
        donor_bin_correct = np.zeros((n_donors, n_bins), dtype=np.float64)
        donor_bin_confidence = np.zeros((n_donors, n_bins), dtype=np.float64)
        np.add.at(donor_bin_count, (donor_code, bin_index), 1.0)
        np.add.at(donor_bin_correct, (donor_code, bin_index), correct)
        np.add.at(donor_bin_confidence, (donor_code, bin_index), confidence)
        boot_bin_count = donor_weights @ donor_bin_count
        boot_bin_correct = donor_weights @ donor_bin_correct
        boot_bin_confidence = donor_weights @ donor_bin_confidence
        boot_bin_accuracy = np.divide(
            boot_bin_correct,
            boot_bin_count,
            out=np.full_like(boot_bin_correct, np.nan),
            where=boot_bin_count > 0,
        )
        boot_bin_mean_conf = np.divide(
            boot_bin_confidence,
            boot_bin_count,
            out=np.zeros_like(boot_bin_confidence),
            where=boot_bin_count > 0,
        )
        boot_ece = np.nansum(
            (boot_bin_count / bootstrap_n[:, None])
            * np.abs(boot_bin_accuracy - boot_bin_mean_conf),
            axis=1,
        )
        point_count = donor_bin_count.sum(axis=0)
        point_accuracy = np.divide(
            donor_bin_correct.sum(axis=0),
            point_count,
            out=np.full(n_bins, np.nan),
            where=point_count > 0,
        )
        point_confidence = np.divide(
            donor_bin_confidence.sum(axis=0),
            point_count,
            out=np.full(n_bins, np.nan),
            where=point_count > 0,
        )
        for bin_number in range(n_bins):
            reliability_records.append(
                {
                    "model": model_name,
                    "bin": bin_number + 1,
                    "lower_edge": bin_number / n_bins,
                    "upper_edge": (bin_number + 1) / n_bins,
                    "n": int(point_count[bin_number]),
                    "mean_confidence": float(point_confidence[bin_number]) if point_count[bin_number] else np.nan,
                    "accuracy": float(point_accuracy[bin_number]) if point_count[bin_number] else np.nan,
                    "accuracy_ci_low": float(np.nanquantile(boot_bin_accuracy[:, bin_number], 0.025)),
                    "accuracy_ci_high": float(np.nanquantile(boot_bin_accuracy[:, bin_number], 0.975)),
                }
            )

        interval_rows.append(
            {
                "model": model_name,
                "macro_f1_point": pooled["macro_f1"],
                "macro_f1_cell_boot_low": float(np.quantile(cell_boot_f1, 0.025)),
                "macro_f1_cell_boot_high": float(np.quantile(cell_boot_f1, 0.975)),
                "macro_f1_donor_boot_low": float(np.quantile(boot_f1, 0.025)),
                "macro_f1_donor_boot_high": float(np.quantile(boot_f1, 0.975)),
                "ece_point": pooled["ece_equal_width_10"],
                "ece_donor_boot_low": float(np.quantile(boot_ece, 0.025)),
                "ece_donor_boot_high": float(np.quantile(boot_ece, 0.975)),
                "brier_point": pooled["multiclass_brier"],
                "brier_donor_boot_low": float(np.quantile(boot_brier, 0.025)),
                "brier_donor_boot_high": float(np.quantile(boot_brier, 0.975)),
                "nll_point": pooled["negative_log_likelihood"],
                "nll_donor_boot_low": float(np.quantile(boot_nll, 0.025)),
                "nll_donor_boot_high": float(np.quantile(boot_nll, 0.975)),
                "n_bootstrap": n_bootstrap,
            }
        )

    pooled_frame = pd.DataFrame(pooled_rows)
    interval_frame = pd.DataFrame(interval_rows)
    reliability_frame = pd.DataFrame(reliability_records)
    pooled_frame.to_csv(destination / "donor_cv_oof_metrics.csv", index=False)
    interval_frame.to_csv(destination / "donor_vs_cell_bootstrap_intervals.csv", index=False)
    reliability_frame.to_csv(destination / "donor_cv_reliability_bins.csv", index=False)

    summary = (
        fold_metrics.groupby("model")
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            ece_mean=("ece_equal_width_10", "mean"),
            ece_sd=("ece_equal_width_10", "std"),
            brier_mean=("multiclass_brier", "mean"),
            brier_sd=("multiclass_brier", "std"),
            nll_mean=("negative_log_likelihood", "mean"),
            nll_sd=("negative_log_likelihood", "std"),
            n_matched_hvgs_mean=("n_matched_hvgs", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(destination / "donor_cv_fold_summary.csv", index=False)

    comparison_pairs = [
        (
            "Matched HVGs plus Logistic Regression",
            "scGPT embeddings plus Logistic Regression",
        ),
        ("Matched HVGs plus Random Forest", "scGPT embeddings plus Random Forest"),
        ("Matched HVGs plus XGBoost", "scGPT embeddings plus XGBoost"),
    ]
    best_classical = pooled_frame[pooled_frame["model"].str.startswith("Matched HVGs")].sort_values("macro_f1").iloc[-1]["model"]
    best_scgpt = pooled_frame[pooled_frame["model"].str.startswith("scGPT embeddings")].sort_values("macro_f1").iloc[-1]["model"]
    comparison_pairs.append((best_classical, best_scgpt))
    paired_rows = []
    seen = set()
    for first, second in comparison_pairs:
        if (first, second) in seen:
            continue
        seen.add((first, second))
        difference = donor_boot_f1[first] - donor_boot_f1[second]
        point = float(
            pooled_frame.set_index("model").loc[first, "macro_f1"]
            - pooled_frame.set_index("model").loc[second, "macro_f1"]
        )
        paired_rows.append(
            {
                "model_a": first,
                "model_b": second,
                "difference_a_minus_b": point,
                "donor_boot_ci_low": float(np.quantile(difference, 0.025)),
                "donor_boot_ci_high": float(np.quantile(difference, 0.975)),
                "probability_difference_gt_zero": float(np.mean(difference > 0)),
                "n_bootstrap": n_bootstrap,
            }
        )
    pd.DataFrame(paired_rows).to_csv(destination / "paired_donor_bootstrap_differences.csv", index=False)

    display_names = {
        "scGPT embeddings plus Logistic Regression": "scGPT emb. + LR",
        "scGPT embeddings plus Random Forest": "scGPT emb. + RF",
        "scGPT embeddings plus XGBoost": "scGPT emb. + XGB",
        "Matched HVGs plus Logistic Regression": "Matched HVGs + LR",
        "Matched HVGs plus Random Forest": "Matched HVGs + RF",
        "Matched HVGs plus XGBoost": "Matched HVGs + XGB",
    }
    model_order = model_names
    palette = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#E8684A", "#9270CA"]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for position, model_name in enumerate(model_order):
        subset = fold_metrics[fold_metrics["model"] == model_name].sort_values("fold")
        jitter = np.linspace(-0.08, 0.08, len(subset))
        ax.scatter(
            np.repeat(position, len(subset)) + jitter,
            subset["macro_f1"],
            s=45,
            color=palette[position],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.errorbar(
            position,
            subset["macro_f1"].mean(),
            yerr=subset["macro_f1"].std(ddof=1),
            fmt="D",
            color="black",
            capsize=5,
            markersize=6,
            zorder=4,
        )
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels([display_names[m] for m in model_order], rotation=22, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Five donor-held-out folds on Indonesia PBMC")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "Figure_DonorHeldOut_Performance.pdf", bbox_inches="tight")
    fig.savefig(destination / "Figure_DonorHeldOut_Performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(13.8, 10.5))
    grid = fig.add_gridspec(4, 3, height_ratios=[3.2, 1.0, 3.2, 1.0], hspace=0.55, wspace=0.33)
    rel = reliability_frame.set_index(["model", "bin"])
    for position, model_name in enumerate(model_order):
        block = position // 3
        column = position % 3
        curve_ax = fig.add_subplot(grid[block * 2, column])
        hist_ax = fig.add_subplot(grid[block * 2 + 1, column], sharex=curve_ax)
        subset = reliability_frame[reliability_frame["model"] == model_name]
        occupied = subset["n"] > 0
        plot = subset[occupied]
        curve_ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
        yerr = np.vstack(
            [
                np.maximum(0.0, plot["accuracy"] - plot["accuracy_ci_low"]),
                np.maximum(0.0, plot["accuracy_ci_high"] - plot["accuracy"]),
            ]
        )
        curve_ax.errorbar(
            plot["mean_confidence"],
            plot["accuracy"],
            yerr=yerr,
            fmt="o-",
            color=palette[position],
            ecolor="#333333",
            capsize=2.5,
            linewidth=1.5,
            markersize=5,
        )
        curve_ax.set_xlim(0, 1)
        curve_ax.set_ylim(0, 1)
        curve_ax.set_title(display_names[model_name], fontsize=10)
        curve_ax.set_ylabel("Observed accuracy")
        curve_ax.grid(alpha=0.18)
        centers = (subset["lower_edge"] + subset["upper_edge"]) / 2
        hist_ax.bar(centers, subset["n"], width=0.09, color=palette[position], alpha=0.75)
        hist_ax.set_yscale("log")
        hist_ax.set_ylabel("n", rotation=0, labelpad=8)
        hist_ax.set_xlabel("Predicted confidence")
        hist_ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Donor-held-out reliability with 95% donor-bootstrap intervals and bin counts",
        fontsize=14,
        y=0.995,
    )
    fig.savefig(destination / "Figure_DonorHeldOut_Reliability.pdf", bbox_inches="tight")
    fig.savefig(destination / "Figure_DonorHeldOut_Reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_cells": expected_n_cells,
        "n_donors": n_donors,
        "n_folds": 5,
        "n_bootstrap": n_bootstrap,
        "best_classical": best_classical,
        "best_scgpt": best_scgpt,
        "pooled_metrics": pooled_rows,
        "paired_differences": paired_rows,
    }
    (destination / "aggregate_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def aggregate_donor_secondary(secondary_root: str, output_root: str) -> dict:
    """Aggregate fold-wise scarcity and calibration results and create figures."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    root = Path(secondary_root)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    scarcity = pd.concat(
        [
            pd.read_csv(root / f"fold_{fold}" / "label_scarcity_metrics.csv")
            for fold in range(5)
        ],
        ignore_index=True,
    )
    temperature = pd.concat(
        [
            pd.read_csv(root / f"fold_{fold}" / "temperature_scaling_metrics.csv")
            for fold in range(5)
        ],
        ignore_index=True,
    )
    scarcity.to_csv(destination / "label_scarcity_all_fits.csv", index=False)
    temperature.to_csv(destination / "temperature_scaling_all_folds.csv", index=False)

    scarcity_fold = (
        scarcity.groupby(
            ["fold", "labels_per_class", "representation", "classifier", "model"],
            as_index=False,
        )
        .agg(
            macro_f1=("macro_f1", "mean"),
            macro_f1_seed_sd=("macro_f1", "std"),
            ece_equal_width_10=("ece_equal_width_10", "mean"),
            multiclass_brier=("multiclass_brier", "mean"),
            negative_log_likelihood=("negative_log_likelihood", "mean"),
        )
    )
    scarcity_fold.to_csv(destination / "label_scarcity_fold_means.csv", index=False)
    scarcity_summary = (
        scarcity_fold.groupby(
            ["labels_per_class", "representation", "classifier", "model"],
            as_index=False,
        )
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_fold_sd=("macro_f1", "std"),
            ece_mean=("ece_equal_width_10", "mean"),
            ece_fold_sd=("ece_equal_width_10", "std"),
            brier_mean=("multiclass_brier", "mean"),
            brier_fold_sd=("multiclass_brier", "std"),
            nll_mean=("negative_log_likelihood", "mean"),
            nll_fold_sd=("negative_log_likelihood", "std"),
        )
    )
    scarcity_summary.to_csv(destination / "label_scarcity_summary.csv", index=False)

    colors = {"scGPT embeddings": "#5B8FF9", "Matched HVGs": "#E8684A"}
    markers = {"scGPT embeddings": "o", "Matched HVGs": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)
    for axis, classifier in zip(axes, ["logreg", "xgboost"]):
        for representation in ["scGPT embeddings", "Matched HVGs"]:
            subset = scarcity_summary[
                (scarcity_summary["classifier"] == classifier)
                & (scarcity_summary["representation"] == representation)
            ].sort_values("labels_per_class")
            axis.errorbar(
                subset["labels_per_class"],
                subset["macro_f1_mean"],
                yerr=subset["macro_f1_fold_sd"],
                marker=markers[representation],
                color=colors[representation],
                capsize=3,
                linewidth=1.8,
                label=representation,
            )
        axis.set_xscale("log")
        axis.set_xticks([10, 25, 50, 100, 250, 500])
        axis.set_xticklabels([10, 25, 50, 100, 250, 500])
        axis.set_xlabel("Labeled training cells per class")
        axis.set_title("Logistic Regression" if classifier == "logreg" else "XGBoost")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Macro F1 on held-out donors")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Donor-held-out label-scarcity analysis (mean ± SD across five folds)")
    fig.tight_layout()
    fig.savefig(destination / "Figure_LabelScarcity.pdf", bbox_inches="tight")
    fig.savefig(destination / "Figure_LabelScarcity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    temperature_summary = (
        temperature.groupby(["pipeline", "calibrated"], as_index=False)
        .agg(
            temperature_mean=("temperature", "mean"),
            temperature_sd=("temperature", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            ece_mean=("ece_equal_width_10", "mean"),
            ece_sd=("ece_equal_width_10", "std"),
            brier_mean=("multiclass_brier", "mean"),
            brier_sd=("multiclass_brier", "std"),
            nll_mean=("negative_log_likelihood", "mean"),
            nll_sd=("negative_log_likelihood", "std"),
        )
    )
    temperature_summary.to_csv(destination / "temperature_scaling_summary.csv", index=False)

    metrics = [
        ("ece_equal_width_10", "ECE"),
        ("multiclass_brier", "Multiclass Brier score"),
        ("negative_log_likelihood", "Negative log-likelihood"),
    ]
    pipelines = temperature["pipeline"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for axis, (metric, label) in zip(axes, metrics):
        for pipeline_index, pipeline in enumerate(pipelines):
            subset = temperature[temperature["pipeline"] == pipeline]
            uncalibrated = subset[~subset["calibrated"]].sort_values("fold")
            calibrated = subset[subset["calibrated"]].sort_values("fold")
            x0 = pipeline_index * 3
            for fold_index in range(5):
                axis.plot(
                    [x0, x0 + 1],
                    [uncalibrated.iloc[fold_index][metric], calibrated.iloc[fold_index][metric]],
                    color=colors["scGPT embeddings" if pipeline.startswith("scGPT") else "Matched HVGs"],
                    alpha=0.35,
                    linewidth=1,
                )
            axis.scatter(
                np.repeat([x0, x0 + 1], 5),
                np.concatenate([uncalibrated[metric].to_numpy(), calibrated[metric].to_numpy()]),
                color=colors["scGPT embeddings" if pipeline.startswith("scGPT") else "Matched HVGs"],
                s=28,
                zorder=3,
            )
        tick_positions = []
        tick_labels = []
        for pipeline_index, pipeline in enumerate(pipelines):
            tick_positions.extend([pipeline_index * 3, pipeline_index * 3 + 1])
            prefix = "scGPT" if pipeline.startswith("scGPT") else "HVG"
            tick_labels.extend([f"{prefix}\nBefore", f"{prefix}\nAfter"])
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Held-out-donor temperature scaling across five outer folds")
    fig.tight_layout()
    fig.savefig(destination / "Figure_TemperatureScaling.pdf", bbox_inches="tight")
    fig.savefig(destination / "Figure_TemperatureScaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_folds": 5,
        "n_scarcity_seeds": int(scarcity["scarcity_seed"].nunique()),
        "label_counts": sorted(int(value) for value in scarcity["labels_per_class"].unique()),
        "temperature_summary": temperature_summary.to_dict(orient="records"),
    }
    (destination / "secondary_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    import numpy as np

    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def aggregate_atlas_benchmark(
    benchmark_root: str,
    output_root: str,
    n_bootstrap: int = 5000,
) -> dict:
    """Aggregate six-atlas matched benchmarks with paired cell bootstraps."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    root = Path(benchmark_root)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(
        [pd.read_csv(root / dataset / "metrics.csv") for dataset in ATLAS_NAMES],
        ignore_index=True,
    )
    metrics.to_csv(destination / "atlas_metrics.csv", index=False)
    model_names = metrics["model"].drop_duplicates().tolist()
    rng = np.random.default_rng(20261003)
    bootstrap_rows = []
    reliability_rows = []
    prediction_cache = {}
    for dataset in ATLAS_NAMES:
        for model_name in model_names:
            data = np.load(
                root / dataset / f"predictions__{_model_slug(model_name)}.npz",
                allow_pickle=False,
            )
            y_true = data["y_true"].astype(np.int64)
            y_prob = data["y_prob"].astype(np.float32)
            y_pred = y_prob.argmax(axis=1)
            class_names = data["class_names"].astype(str)
            prediction_cache[(dataset, model_name)] = (y_true, y_pred, y_prob, class_names)
            n_classes = len(class_names)
            confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
            np.add.at(confusion, (y_true, y_pred), 1)
            draws = rng.multinomial(
                len(y_true),
                confusion.ravel() / confusion.sum(),
                size=n_bootstrap,
            ).reshape(n_bootstrap, n_classes, n_classes)
            boot_f1 = _macro_f1_from_confusions(draws)
            point_f1 = float(
                metrics.loc[
                    (metrics["dataset"] == dataset) & (metrics["model"] == model_name),
                    "macro_f1",
                ].iloc[0]
            )
            bootstrap_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "macro_f1": point_f1,
                    "cell_boot_ci_low": float(np.quantile(boot_f1, 0.025)),
                    "cell_boot_ci_high": float(np.quantile(boot_f1, 0.975)),
                    "n_test_cells": len(y_true),
                    "n_bootstrap": n_bootstrap,
                }
            )

            confidence = y_prob.max(axis=1).astype(np.float64)
            correct = (y_pred == y_true)
            bin_index = np.minimum((confidence * 10).astype(np.int64), 9)
            for bin_number in range(10):
                mask = bin_index == bin_number
                total = int(mask.sum())
                successes = int(correct[mask].sum()) if total else 0
                low, high = _wilson_interval(successes, total)
                reliability_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "bin": bin_number + 1,
                        "lower_edge": bin_number / 10,
                        "upper_edge": (bin_number + 1) / 10,
                        "n": total,
                        "mean_confidence": float(confidence[mask].mean()) if total else np.nan,
                        "accuracy": successes / total if total else np.nan,
                        "accuracy_wilson_low": low,
                        "accuracy_wilson_high": high,
                    }
                )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    reliability_frame = pd.DataFrame(reliability_rows)
    bootstrap_frame.to_csv(destination / "atlas_cell_bootstrap_intervals.csv", index=False)
    reliability_frame.to_csv(destination / "atlas_reliability_bins.csv", index=False)

    paired_rows = []
    classifier_pairs = [
        (
            "Matched HVGs plus Logistic Regression",
            "scGPT embeddings plus Logistic Regression",
            "Logistic Regression",
        ),
        (
            "Matched HVGs plus Random Forest",
            "scGPT embeddings plus Random Forest",
            "Random Forest",
        ),
        (
            "Matched HVGs plus XGBoost",
            "scGPT embeddings plus XGBoost",
            "XGBoost",
        ),
    ]
    for dataset in ATLAS_NAMES:
        dataset_metrics = metrics[metrics["dataset"] == dataset]
        best_matched = dataset_metrics[
            dataset_metrics["representation"] == "matched_hvgs"
        ].sort_values("macro_f1").iloc[-1]["model"]
        best_scgpt = dataset_metrics[
            dataset_metrics["representation"] == "scgpt_embeddings"
        ].sort_values("macro_f1").iloc[-1]["model"]
        pairs = classifier_pairs + [(best_matched, best_scgpt, "Best observed pipeline")]
        seen = set()
        for model_a, model_b, comparison in pairs:
            if (model_a, model_b) in seen:
                continue
            seen.add((model_a, model_b))
            y_true_a, pred_a, _, classes_a = prediction_cache[(dataset, model_a)]
            y_true_b, pred_b, _, classes_b = prediction_cache[(dataset, model_b)]
            if not np.array_equal(y_true_a, y_true_b) or not np.array_equal(classes_a, classes_b):
                raise AssertionError(f"Prediction alignment failed for {dataset}")
            n_classes = len(classes_a)
            state = (y_true_a * n_classes + pred_a) * n_classes + pred_b
            state_counts = np.bincount(state, minlength=n_classes ** 3)
            joint_draws = rng.multinomial(
                len(y_true_a),
                state_counts / state_counts.sum(),
                size=n_bootstrap,
            ).reshape(n_bootstrap, n_classes, n_classes, n_classes)
            f1_a = _macro_f1_from_confusions(joint_draws.sum(axis=3))
            f1_b = _macro_f1_from_confusions(joint_draws.sum(axis=2))
            difference = f1_a - f1_b
            point_a = float(
                dataset_metrics.loc[dataset_metrics["model"] == model_a, "macro_f1"].iloc[0]
            )
            point_b = float(
                dataset_metrics.loc[dataset_metrics["model"] == model_b, "macro_f1"].iloc[0]
            )
            paired_rows.append(
                {
                    "dataset": dataset,
                    "comparison": comparison,
                    "model_a": model_a,
                    "model_b": model_b,
                    "difference_a_minus_b": point_a - point_b,
                    "paired_cell_boot_ci_low": float(np.quantile(difference, 0.025)),
                    "paired_cell_boot_ci_high": float(np.quantile(difference, 0.975)),
                    "probability_difference_gt_zero": float(np.mean(difference > 0)),
                    "n_bootstrap": n_bootstrap,
                }
            )
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(destination / "atlas_paired_cell_bootstrap_differences.csv", index=False)

    display_datasets = {
        "TNBC_Breast_Cancer": "TNBC",
        "Indonesia_PBMC": "Indonesia PBMC",
        "Brain_Atlas": "Brain atlas",
        "Multi_Tissue_TME": "Multi-tissue TME",
        "Human_Pancreas": "Human pancreas",
        "Pig_Pancreas": "Pig pancreas",
    }
    display_models = {
        "scGPT embeddings plus Logistic Regression": "scGPT emb. + LR",
        "scGPT embeddings plus Random Forest": "scGPT emb. + RF",
        "scGPT embeddings plus XGBoost": "scGPT emb. + XGB",
        "Matched HVGs plus Logistic Regression": "Matched HVGs + LR",
        "Matched HVGs plus Random Forest": "Matched HVGs + RF",
        "Matched HVGs plus XGBoost": "Matched HVGs + XGB",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.8))
    for axis, metric, title, format_string in [
        (axes[0], "macro_f1", "Macro F1", ".3f"),
        (axes[1], "ece_equal_width_10", "Expected calibration error", ".3f"),
    ]:
        matrix = (
            metrics.pivot(index="dataset", columns="model", values=metric)
            .reindex(index=ATLAS_NAMES, columns=model_names)
        )
        image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis_r" if metric.startswith("ece") else "viridis")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                rgba = image.cmap(image.norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column,
                    row,
                    format(value, format_string),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="black" if luminance > 0.55 else "white",
                )
        axis.set_xticks(range(len(model_names)))
        axis.set_xticklabels([display_models[value] for value in model_names], rotation=30, ha="right")
        axis.set_yticks(range(len(ATLAS_NAMES)))
        axis.set_yticklabels([display_datasets[value] for value in ATLAS_NAMES])
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    fig.suptitle("Matched within-atlas benchmark across six single-cell atlases", fontsize=14)
    fig.tight_layout()
    fig.savefig(destination / "Figure_AtlasBenchmark.pdf", bbox_inches="tight")
    fig.savefig(destination / "Figure_AtlasBenchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    colors = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#E8684A", "#9270CA"]
    for dataset_index, dataset in enumerate(ATLAS_NAMES):
        fig = plt.figure(figsize=(13.8, 10.5))
        grid = fig.add_gridspec(4, 3, height_ratios=[3.2, 1.0, 3.2, 1.0], hspace=0.55, wspace=0.33)
        for position, model_name in enumerate(model_names):
            block = position // 3
            column = position % 3
            curve_axis = fig.add_subplot(grid[block * 2, column])
            count_axis = fig.add_subplot(grid[block * 2 + 1, column], sharex=curve_axis)
            subset = reliability_frame[
                (reliability_frame["dataset"] == dataset)
                & (reliability_frame["model"] == model_name)
            ]
            occupied = subset[subset["n"] > 0]
            curve_axis.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
            y_error = np.vstack(
                [
                    np.maximum(
                        0.0,
                        occupied["accuracy"] - occupied["accuracy_wilson_low"],
                    ),
                    np.maximum(
                        0.0,
                        occupied["accuracy_wilson_high"] - occupied["accuracy"],
                    ),
                ]
            )
            curve_axis.errorbar(
                occupied["mean_confidence"],
                occupied["accuracy"],
                yerr=y_error,
                fmt="o-",
                color=colors[position],
                ecolor="#333333",
                capsize=2.5,
                linewidth=1.5,
                markersize=5,
            )
            curve_axis.set_xlim(0, 1)
            curve_axis.set_ylim(0, 1)
            curve_axis.set_title(display_models[model_name], fontsize=10)
            curve_axis.set_ylabel("Observed accuracy")
            curve_axis.grid(alpha=0.18)
            centers = (subset["lower_edge"] + subset["upper_edge"]) / 2
            count_axis.bar(centers, subset["n"], width=0.09, color=colors[position], alpha=0.75)
            count_axis.set_yscale("log")
            count_axis.set_ylabel("n", rotation=0, labelpad=8)
            count_axis.set_xlabel("Predicted confidence")
            count_axis.grid(axis="y", alpha=0.15)
        fig.suptitle(
            f"{display_datasets[dataset]} reliability with 95% Wilson intervals and bin counts",
            fontsize=14,
            y=0.995,
        )
        fig.savefig(destination / f"Reliability_{dataset}.pdf", bbox_inches="tight")
        fig.savefig(destination / f"Reliability_{dataset}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    payload = {
        "datasets": list(ATLAS_NAMES),
        "n_datasets": len(ATLAS_NAMES),
        "n_models": len(model_names),
        "n_total_cells": int(
            metrics.drop_duplicates("dataset")["n_cells"].sum()
        ),
        "n_bootstrap": n_bootstrap,
        "metrics": metrics.to_dict(orient="records"),
        "paired_differences": paired_rows,
    }
    (destination / "atlas_aggregate_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload
