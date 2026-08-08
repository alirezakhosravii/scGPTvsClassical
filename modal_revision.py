"""Modal jobs for the reviewer-requested scGPT benchmark revision.

This module keeps all large public datasets, pretrained weights, cached
embeddings, and outputs in a dedicated Modal volume.  The first stage only
downloads and audits metadata; model fitting is added after the biological
grouping variable has been verified.
"""

from __future__ import annotations

import json
import hashlib
import shutil
import sys
from pathlib import Path

import modal


app = modal.App("scgpt-calibration-revision")
volume = modal.Volume.from_name("scgpt-calibration-revision", create_if_missing=True)

audit_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "anndata==0.12.3",
    "h5py>=3.11",
    "pandas>=2.2",
    "requests>=2.32",
    "scipy==1.13.1",
    "gdown>=5.2",
)

_HERE = Path(__file__).resolve().parent
_SCGPT_SOURCE_LOCAL = _HERE.parent / "reference_scGPT"
_REVISION_ANALYSIS_LOCAL = _HERE / "revision_analysis.py"

benchmark_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install(
        "anndata==0.12.3",
        "datasets==2.21.0",
        "h5py>=3.11",
        "ipython>=8.20",
        "matplotlib>=3.8,<4",
        "networkx>=3.2",
        "numba>=0.60",
        "numpy==1.26.4",
        "pandas>=2.2,<3",
        "scanpy>=1.10,<2",
        "scikit-learn>=1.6,<2",
        "scikit-misc>=0.3",
        "scipy==1.13.1",
        "seaborn>=0.13",
        "tqdm>=4.66",
        "typing-extensions>=4.12",
        "xgboost>=2.1,<4",
        "ninja>=1.11",
        "packaging>=24",
        "psutil>=6",
    )
    .env({"MAX_JOBS": "8"})
    .run_commands("pip install flash-attn==2.8.3 --no-build-isolation")
    .add_local_dir(str(_SCGPT_SOURCE_LOCAL), remote_path="/root/scgpt_src")
    .add_local_file(str(_REVISION_ANALYSIS_LOCAL), remote_path="/root/revision_analysis.py")
)

DATASETS = {
    "TNBC_Breast_Cancer": "https://datasets.cellxgene.cziscience.com/af8c4fce-4c63-4671-b339-91a383cf36f6.h5ad",
    "Indonesia_PBMC": "https://datasets.cellxgene.cziscience.com/665714af-4be5-49a3-913b-5ab5ac25620d.h5ad",
    "Brain_Atlas": "https://datasets.cellxgene.cziscience.com/0ab54d91-066c-4223-a9ea-6a3b0d1adef4.h5ad",
    "Multi_Tissue_TME": "https://datasets.cellxgene.cziscience.com/921d46a3-69b4-44a8-b2d6-9ef5c7803bc3.h5ad",
    "Human_Pancreas": "https://datasets.cellxgene.cziscience.com/00d88707-e33a-4c2a-821a-cdc32a98d050.h5ad",
    "Pig_Pancreas": "https://datasets.cellxgene.cziscience.com/55cfae87-6348-44df-a4ed-c132569dea54.h5ad",
}

SCGPT_GDRIVE = "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing"
SCGPT_CHECKPOINT_FILES = ("args.json", "vocab.json", "best_model.pt")


def _download_streaming(url: str, destination: Path) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 600)) as response:
        if response.status_code not in (200, 206):
            response.raise_for_status()
        if existing and response.status_code == 200:
            existing = 0
            mode = "wb"
        else:
            mode = "ab" if existing else "wb"
        total_header = response.headers.get("content-length")
        expected = existing + int(total_header) if total_header else None
        downloaded = existing
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                downloaded += len(chunk)
                if downloaded % (512 * 1024 * 1024) < len(chunk):
                    print(f"downloaded {downloaded / 1e9:.2f} GB", flush=True)
    if expected is not None and downloaded != expected:
        raise RuntimeError(f"download incomplete: {downloaded} bytes, expected {expected}")
    partial.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=audit_image,
    volumes={"/vol": volume},
    cpu=4.0,
    memory=8192,
    timeout=21600,
)
def prepare_scgpt_checkpoint() -> str:
    import gdown

    model_dir = Path("/vol/models/scGPT_human")
    model_dir.mkdir(parents=True, exist_ok=True)
    if not all((model_dir / name).exists() for name in SCGPT_CHECKPOINT_FILES):
        download_dir = Path("/vol/models/gdrive_download")
        download_dir.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            SCGPT_GDRIVE,
            output=str(download_dir),
            quiet=False,
            use_cookies=False,
        )
        candidate_parents = []
        for model_path in download_dir.rglob("best_model.pt"):
            parent = model_path.parent
            if all((parent / name).exists() for name in SCGPT_CHECKPOINT_FILES):
                candidate_parents.append(parent)
        if not candidate_parents:
            raise FileNotFoundError("Could not locate a complete scGPT checkpoint after Google Drive download")
        source_dir = sorted(candidate_parents, key=lambda p: len(p.parts))[0]
        for name in SCGPT_CHECKPOINT_FILES:
            shutil.copy2(source_dir / name, model_dir / name)
        volume.commit()

    payload = {
        "model_dir": str(model_dir),
        "files": {
            name: {
                "size_bytes": int((model_dir / name).stat().st_size),
                "sha256": _sha256(model_dir / name),
            }
            for name in SCGPT_CHECKPOINT_FILES
        },
    }
    audit_path = Path("/vol/audit/scgpt_checkpoint.json")
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=audit_image,
    volumes={"/vol": volume},
    cpu=4.0,
    memory=16384,
    timeout=21600,
)
def audit_indonesia_metadata() -> str:
    import anndata as ad
    import pandas as pd

    root = Path("/vol")
    data_path = root / "data" / "Indonesia_PBMC.h5ad"
    if not data_path.exists():
        _download_streaming(DATASETS["Indonesia_PBMC"], data_path)
        volume.commit()

    adata = ad.read_h5ad(data_path, backed="r")
    obs = adata.obs
    candidate_tokens = (
        "donor",
        "sample",
        "batch",
        "individual",
        "patient",
        "subject",
        "specimen",
    )
    candidates = [
        column
        for column in obs.columns
        if any(token in str(column).lower() for token in candidate_tokens)
    ]
    selected = sorted(set(candidates + [c for c in ["cell_type", "tissue", "assay"] if c in obs]))
    summaries: dict[str, dict] = {}
    for column in selected:
        values = obs[column]
        counts = values.astype("string").value_counts(dropna=False)
        summaries[str(column)] = {
            "dtype": str(values.dtype),
            "n_unique": int(values.nunique(dropna=True)),
            "n_missing": int(values.isna().sum()),
            "top_values": {str(k): int(v) for k, v in counts.head(12).items()},
        }
    payload = {
        "path": str(data_path),
        "file_size_bytes": data_path.stat().st_size,
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "obs_columns": [str(c) for c in obs.columns],
        "candidate_summaries": summaries,
        "var_columns": [str(c) for c in adata.var.columns],
    }
    output_path = root / "audit" / "indonesia_metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=audit_image,
    volumes={"/vol": volume},
    cpu=4.0,
    memory=16384,
    timeout=21600,
)
def prepare_and_audit_dataset(dataset_name: str) -> str:
    import anndata as ad

    if dataset_name not in DATASETS:
        raise KeyError(dataset_name)
    data_path = Path("/vol/data") / f"{dataset_name}.h5ad"
    if not data_path.exists():
        print(f"[{dataset_name}] downloading", flush=True)
        _download_streaming(DATASETS[dataset_name], data_path)
        volume.commit()
    adata = ad.read_h5ad(data_path, backed="r")
    if "cell_type" not in adata.obs:
        raise KeyError(f"{dataset_name} lacks obs['cell_type']")
    class_counts = adata.obs["cell_type"].astype(str).value_counts(dropna=False)
    payload = {
        "dataset": dataset_name,
        "path": str(data_path),
        "file_size_bytes": int(data_path.stat().st_size),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "raw_present": adata.raw is not None,
        "raw_shape": [int(adata.raw.n_obs), int(adata.raw.n_vars)] if adata.raw is not None else None,
        "n_cell_types": int(len(class_counts)),
        "n_cell_types_more_than_five_cells": int((class_counts > 5).sum()),
        "cell_type_counts": {str(key): int(value) for key, value in class_counts.items()},
        "feature_name_present": bool("feature_name" in adata.var.columns),
        "obs_columns": [str(column) for column in adata.obs.columns],
        "var_columns": [str(column) for column in adata.var.columns],
    }
    audit_path = Path("/vol/audit/datasets") / f"{dataset_name}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=audit_image,
    volumes={"/vol": volume},
    cpu=4.0,
    memory=16384,
    timeout=3600,
)
def audit_indonesia_matrix() -> str:
    import anndata as ad
    import numpy as np
    from scipy import sparse

    data_path = Path("/vol/data/Indonesia_PBMC.h5ad")
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    adata = ad.read_h5ad(data_path, backed="r")
    # Contiguous backed slices avoid an anndata/scipy fancy-indexing
    # incompatibility while remaining sufficient to identify count scaling.
    sample = adata.X[:256, :]
    values = sample.data if sparse.issparse(sample) else np.asarray(sample).ravel()
    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    raw_summary = None
    if adata.raw is not None:
        raw_sample = adata.raw.X[:256, :]
        raw_values = raw_sample.data if sparse.issparse(raw_sample) else np.asarray(raw_sample).ravel()
        raw_values = np.asarray(raw_values, dtype=float)
        raw_positive = raw_values[raw_values > 0]
        raw_summary = {
            "shape": [int(adata.raw.n_obs), int(adata.raw.n_vars)],
            "x_type": type(adata.raw.X).__name__,
            "sample_nonzero_values": int(len(raw_positive)),
            "sample_min_positive": float(raw_positive.min()) if len(raw_positive) else None,
            "sample_max": float(raw_values.max()) if len(raw_values) else None,
            "sample_fraction_noninteger": float(np.mean(np.abs(raw_positive - np.rint(raw_positive)) > 1e-7)) if len(raw_positive) else None,
            "sample_row_sums_quantiles": {
                str(q): float(np.quantile(np.asarray(raw_sample.sum(axis=1)).ravel(), q))
                for q in [0.0, 0.25, 0.5, 0.75, 1.0]
            },
        }
    payload = {
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "x_type": type(adata.X).__name__,
        "layers": [str(k) for k in adata.layers.keys()],
        "raw_present": adata.raw is not None,
        "uns_keys": [str(k) for k in adata.uns.keys()],
        "sample_nonzero_values": int(len(positive)),
        "sample_min_positive": float(positive.min()) if len(positive) else None,
        "sample_max": float(values.max()) if len(values) else None,
        "sample_fraction_noninteger": float(np.mean(np.abs(positive - np.rint(positive)) > 1e-7)) if len(positive) else None,
        "sample_quantiles_positive": {
            str(q): float(np.quantile(positive, q))
            for q in [0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
        } if len(positive) else {},
        "sample_row_sums_quantiles": {
            str(q): float(np.quantile(np.asarray(sample.sum(axis=1)).ravel(), q))
            for q in [0.0, 0.25, 0.5, 0.75, 1.0]
        },
        "raw_summary": raw_summary,
    }
    output_path = Path("/vol/audit/indonesia_matrix.json")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=audit_image,
    volumes={"/vol": volume},
    cpu=4.0,
    memory=32768,
    timeout=3600,
)
def audit_atlas_expression_preprocessing(dataset_name: str) -> str:
    """Compare supplied X with normalize_total(1e4)+log1p reconstructed from raw."""
    import anndata as ad
    import numpy as np
    from scipy import sparse

    if dataset_name not in DATASETS:
        raise KeyError(dataset_name)
    path = Path("/vol/data") / f"{dataset_name}.h5ad"
    adata = ad.read_h5ad(path, backed="r")
    if adata.raw is None or adata.raw.n_vars != adata.n_vars:
        raise ValueError(f"Raw alignment unavailable for {dataset_name}")
    n = min(512, adata.n_obs)
    supplied = adata.X[:n, :]
    raw = adata.raw.X[:n, :]
    supplied = supplied.toarray() if sparse.issparse(supplied) else np.asarray(supplied)
    raw = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
    row_sums = raw.sum(axis=1, keepdims=True)
    normalized = np.log1p(raw * np.divide(10000.0, row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums > 0))
    difference = np.abs(np.asarray(supplied, dtype=float) - normalized)
    payload = {
        "dataset": dataset_name,
        "n_cells": n,
        "n_genes": int(adata.n_vars),
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "fraction_with_difference_gt_1e_6": float(np.mean(difference > 1e-6)),
        "allclose_rtol_1e_5_atol_1e_6": bool(np.allclose(supplied, normalized, rtol=1e-5, atol=1e-6)),
        "supplied_row_sum_median": float(np.median(supplied.sum(axis=1))),
        "reconstructed_row_sum_median": float(np.median(normalized.sum(axis=1))),
    }
    output = Path("/vol/audit/atlas_expression_preprocessing") / f"{dataset_name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    gpu="H100",
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
def smoke_test_gpu_stack() -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import smoke_test_scgpt

    payload = smoke_test_scgpt(
        data_path="/vol/data/Indonesia_PBMC.h5ad",
        model_dir="/vol/models/scGPT_human",
        output_path="/vol/audit/scgpt_gpu_smoke.json",
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    gpu="H100",
    cpu=16.0,
    memory=65536,
    timeout=7200,
)
def debug_donor_fold_gpu() -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import run_donor_fold

    payload = run_donor_fold(
        fold=0,
        data_path="/vol/data/Indonesia_PBMC.h5ad",
        model_dir="/vol/models/scGPT_human",
        output_root="/vol/results/donor_cv_debug",
        debug_cap_per_partition=5000,
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    gpu="H100",
    cpu=16.0,
    memory=65536,
    timeout=21600,
)
def run_donor_fold_gpu(fold: int) -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import run_donor_fold

    payload = run_donor_fold(
        fold=fold,
        data_path="/vol/data/Indonesia_PBMC.h5ad",
        model_dir="/vol/models/scGPT_human",
        output_root="/vol/results/donor_cv",
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    cpu=16.0,
    memory=65536,
    timeout=21600,
)
def aggregate_donor_cv_cpu() -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import aggregate_donor_cv

    payload = aggregate_donor_cv(
        fold_root="/vol/results/donor_cv",
        output_root="/vol/results/donor_cv_aggregate",
        expected_n_cells=462034,
        n_bootstrap=5000,
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    gpu="H100",
    cpu=16.0,
    memory=65536,
    timeout=21600,
)
def run_donor_secondary_fold_gpu(fold: int) -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import run_donor_secondary_fold

    payload = run_donor_secondary_fold(
        fold=fold,
        data_path="/vol/data/Indonesia_PBMC.h5ad",
        fold_root="/vol/results/donor_cv",
        output_root="/vol/results/donor_cv_secondary",
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    cpu=16.0,
    memory=65536,
    timeout=21600,
)
def aggregate_donor_secondary_cpu() -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import aggregate_donor_secondary

    payload = aggregate_donor_secondary(
        secondary_root="/vol/results/donor_cv_secondary",
        output_root="/vol/results/donor_cv_secondary_aggregate",
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    gpu="H100",
    cpu=16.0,
    memory=65536,
    timeout=21600,
)
def run_atlas_benchmark_gpu(dataset_name: str) -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import run_atlas_cell_split

    payload = run_atlas_cell_split(
        dataset_name=dataset_name,
        data_path=f"/vol/data/{dataset_name}.h5ad",
        model_dir="/vol/models/scGPT_human",
        output_root="/vol/results/atlas_benchmark_final_v2",
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    cpu=16.0,
    memory=65536,
    timeout=21600,
)
def aggregate_atlas_benchmark_cpu() -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import aggregate_atlas_benchmark

    payload = aggregate_atlas_benchmark(
        benchmark_root="/vol/results/atlas_benchmark_final_v2",
        output_root="/vol/results/atlas_benchmark_final_v2_aggregate",
        n_bootstrap=5000,
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
def audit_indonesia_expression_preprocessing() -> str:
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp

    sys.path.insert(0, "/root")
    from revision_analysis import _raw_slice_to_adata

    adata = ad.read_h5ad("/vol/data/Indonesia_PBMC.h5ad")
    genes = json.loads(
        Path("/vol/results/donor_cv/fold_0/selected_genes.json").read_text(encoding="utf-8")
    )["var_names"]
    observation_index = np.arange(2048, dtype=np.int64)
    derived = _raw_slice_to_adata(adata, observation_index, genes)
    sc.pp.normalize_total(derived, target_sum=1e4)
    sc.pp.log1p(derived)
    expected = derived.X.toarray() if sp.issparse(derived.X) else np.asarray(derived.X)
    supplied = adata[observation_index, genes].X
    supplied = supplied.toarray() if sp.issparse(supplied) else np.asarray(supplied)
    difference = np.abs(expected.astype(np.float64) - supplied.astype(np.float64))
    payload = {
        "n_cells": len(observation_index),
        "n_genes": len(genes),
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "fraction_with_difference_gt_1e_6": float(np.mean(difference > 1e-6)),
        "allclose_rtol_1e_5_atol_1e_6": bool(np.allclose(expected, supplied, rtol=1e-5, atol=1e-6)),
    }
    path = Path("/vol/audit/indonesia_expression_preprocessing.json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    cpu=2.0,
    memory=8192,
    timeout=1800,
)
def audit_software_environment() -> str:
    import importlib.metadata as metadata
    import platform

    distributions = [
        "anndata",
        "flash-attn",
        "matplotlib",
        "numpy",
        "pandas",
        "scanpy",
        "scikit-learn",
        "scikit-misc",
        "scipy",
        "seaborn",
        "torch",
        "xgboost",
    ]
    versions = {}
    for name in distributions:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "scgpt_version": "0.2.5",
        "scgpt_source_commit": "cebd6fa",
    }
    path = Path("/vol/audit/software_environment.json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    gpu="H100",
    cpu=16.0,
    memory=65536,
    timeout=21600,
)
def rerun_normalized_donor_fold_gpu(fold: int) -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import rerun_donor_fold_with_raw_normalized_expression

    payload = rerun_donor_fold_with_raw_normalized_expression(
        fold=fold,
        data_path="/vol/data/Indonesia_PBMC.h5ad",
        cache_root="/vol/results/donor_cv",
        output_root="/vol/results/donor_cv_normalized_final_v2",
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    gpu="H100",
    cpu=16.0,
    memory=65536,
    timeout=21600,
)
def run_normalized_donor_secondary_fold_gpu(fold: int) -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import run_donor_secondary_fold

    payload = run_donor_secondary_fold(
        fold=fold,
        data_path="/vol/data/Indonesia_PBMC.h5ad",
        fold_root="/vol/results/donor_cv",
        output_root="/vol/results/donor_cv_secondary_normalized_final_v2",
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    cpu=16.0,
    memory=65536,
    timeout=21600,
)
def aggregate_normalized_donor_cv_cpu() -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import aggregate_donor_cv

    payload = aggregate_donor_cv(
        fold_root="/vol/results/donor_cv_normalized_final_v2",
        output_root="/vol/results/donor_cv_normalized_final_v2_aggregate",
        expected_n_cells=462034,
        n_bootstrap=5000,
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.function(
    image=benchmark_image,
    volumes={"/vol": volume},
    cpu=16.0,
    memory=65536,
    timeout=21600,
)
def aggregate_normalized_donor_secondary_cpu() -> str:
    sys.path.insert(0, "/root")
    from revision_analysis import aggregate_donor_secondary

    payload = aggregate_donor_secondary(
        secondary_root="/vol/results/donor_cv_secondary_normalized_final_v2",
        output_root="/vol/results/donor_cv_secondary_normalized_final_v2_aggregate",
    )
    volume.commit()
    return json.dumps(payload, indent=2)


@app.local_entrypoint()
def main(stage: str = "metadata", dataset: str = "Brain_Atlas", fold: int = 0):
    if stage == "metadata":
        print(audit_indonesia_metadata.remote())
    elif stage == "matrix":
        print(audit_indonesia_matrix.remote())
    elif stage == "prepare-all-datasets":
        for result in prepare_and_audit_dataset.map(DATASETS.keys(), order_outputs=True):
            print(result)
    elif stage == "checkpoint":
        print(prepare_scgpt_checkpoint.remote())
    elif stage == "smoke":
        print(smoke_test_gpu_stack.remote())
    elif stage == "debug-fold":
        print(debug_donor_fold_gpu.remote())
    elif stage == "donor-folds":
        for result in run_donor_fold_gpu.map(range(5), order_outputs=True):
            print(result)
    elif stage == "aggregate-donor-cv":
        print(aggregate_donor_cv_cpu.remote())
    elif stage == "donor-secondary-folds":
        for result in run_donor_secondary_fold_gpu.map(range(5), order_outputs=True):
            print(result)
    elif stage == "aggregate-donor-secondary":
        print(aggregate_donor_secondary_cpu.remote())
    elif stage == "atlas-benchmarks":
        for result in run_atlas_benchmark_gpu.map(DATASETS.keys(), order_outputs=True):
            print(result)
    elif stage == "atlas-one":
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        print(run_atlas_benchmark_gpu.remote(dataset))
    elif stage == "aggregate-atlas-benchmarks":
        print(aggregate_atlas_benchmark_cpu.remote())
    elif stage == "audit-indonesia-expression":
        print(audit_indonesia_expression_preprocessing.remote())
    elif stage == "audit-atlas-expression":
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        print(audit_atlas_expression_preprocessing.remote(dataset))
    elif stage == "audit-environment":
        print(audit_software_environment.remote())
    elif stage == "normalized-donor-folds":
        for result in rerun_normalized_donor_fold_gpu.map(range(5), order_outputs=True):
            print(result)
    elif stage == "normalized-donor-one":
        if fold not in range(5):
            raise ValueError(f"Unknown fold: {fold}")
        print(rerun_normalized_donor_fold_gpu.remote(fold))
    elif stage == "aggregate-normalized-donor-cv":
        print(aggregate_normalized_donor_cv_cpu.remote())
    elif stage == "normalized-donor-secondary-folds":
        for result in run_normalized_donor_secondary_fold_gpu.map(range(5), order_outputs=True):
            print(result)
    elif stage == "normalized-donor-secondary-one":
        if fold not in range(5):
            raise ValueError(f"Unknown fold: {fold}")
        print(run_normalized_donor_secondary_fold_gpu.remote(fold))
    elif stage == "aggregate-normalized-donor-secondary":
        print(aggregate_normalized_donor_secondary_cpu.remote())
    else:
        raise ValueError(f"Unknown stage: {stage}")
