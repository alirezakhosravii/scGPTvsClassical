# scGPTvsClassical

Reproducibility repository for:

> **An Accuracy and Calibration Benchmark of scGPT Pipelines and Matched Classical Models for Cell Type Annotation**

The final revision snapshot is in `paper/final_submission/`, `results/final_submission/`, `analysis/final_submission/`, and `computational_audit/final_submission/`. Earlier root-level and `revision/` materials are retained only as provenance and are superseded by this snapshot.

## Final benchmark design

- Three complete human atlases evaluated with five donor-held-out outer folds: TNBC, Indonesia PBMC, and brain.
- 965,427 aligned out-of-fold cells from 312 donor identifiers.
- Frozen scGPT 0.2.5 whole-human embeddings compared with normalized expression of the same training-selected genes.
- A training-fitted SVD-512 representation used as a dimensionality control where five complete folds were available.
- Logistic Regression, Random Forest, and XGBoost tuned separately for every representation using only outer-training donors.
- Primary comparisons fixed the downstream classifier; outer-test performance was never used to select a winning head.
- Paired 5,000-draw donor-cluster bootstraps, calibration metrics, held-out-donor temperature scaling, strict PBMC label scarcity, and class-level error analyses.
- Pig pancreas reported separately as a random-split cross-species stress test without an orthology map.

## Results at a glance

Matched expression had higher pooled out-of-fold Macro F1 in all three TNBC and all three PBMC classifier-matched comparisons. Frozen scGPT embeddings were higher in all three brain comparisons. Donor-cluster intervals supported all six TNBC/PBMC expression advantages and the brain Random Forest scGPT advantage; the brain Logistic Regression and XGBoost intervals crossed zero.

| Atlas | Classifier | scGPT embeddings | Matched expression | SVD-512 |
|---|---|---:|---:|---:|
| TNBC | Logistic Regression | 0.983 | 0.990 | 0.989 |
| TNBC | Random Forest | 0.968 | 0.985 | 0.982 |
| TNBC | XGBoost | 0.978 | 0.990 | 0.988 |
| PBMC | Logistic Regression | 0.949 | 0.959 | 0.958 |
| PBMC | Random Forest | 0.915 | 0.933 | — |
| PBMC | XGBoost | 0.935 | 0.957 | — |
| Brain | Logistic Regression | 0.940 | 0.911 | 0.906 |
| Brain | Random Forest | 0.908 | 0.870 | 0.848 |
| Brain | XGBoost | 0.930 | 0.925 | 0.904 |

The evidence is deliberately scoped to the tested frozen-scGPT pipeline in within-atlas unseen-donor settings. It does not evaluate official scGPT fine-tuning, newer checkpoints, other foundation models, scVI-style generative models, zero-shot discovery, or independently annotated cross-cohort transfer.

## Repository map

| Path | Purpose |
|---|---|
| `paper/final_submission/` | Clean and Track Changes Word manuscripts, proof PDFs, Supplementary Information, point-by-point response, dated cover letter, and unified LaTeX source. |
| `results/final_submission/` | Compact aggregate CSV/JSON files underlying every reported result and figure. |
| `analysis/final_submission/` | Donor-held-out analysis, aggregation, figure/table generation, and environment requirements. |
| `computational_audit/final_submission/` | Donor-group audit, dataset citations, software versions, checkpoint hashes, pipeline coverage, and computational scope. |

## Reproducing reported tables and figures without cloud compute

The supplied compact result tables are sufficient to regenerate all manuscript tables and figures locally:

```bash
python analysis/final_submission/build_final_artifacts.py \
  results/final_submission \
  paper/final_submission/source
```

The Modal launcher and full fold implementation are preserved for auditability, but complete embedding extraction and model fitting are optional and incur substantial cloud cost. They are not required to inspect or compile the final submission.

## Environment and checkpoint

The completed runs used Python 3.11.10, scGPT 0.2.5 at source commit `cebd6fa`, PyTorch 2.5.1/CUDA 12.4, Scanpy 1.11.5, scikit-learn 1.9.0, XGBoost 3.2.0, and NVIDIA H100 80 GB GPUs. Exact software and checkpoint records are in `computational_audit/final_submission/`.

## License and contact

Repository code is released under the [MIT License](LICENSE). Public datasets and scGPT weights remain subject to their original licences. For questions, open an issue or contact Anna Supernat at `abednarz@gumed.edu.pl`.
