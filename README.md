# scGPTvsClassical

Reproducibility repository for:

> **An Accuracy and Calibration Benchmark of scGPT Pipelines and Matched Classical Models for Cell Type Annotation**

Permanent revision snapshot: `scientific-reports-revision-2026-08-09`.

This revision compares official frozen scGPT cell embeddings with normalized expression of the same training-selected, scGPT-vocabulary-matched genes. Logistic Regression, Random Forest, and XGBoost are paired with both representations across six public atlases.

> [!IMPORTANT]
> The corrected revision record is defined by `revision_analysis.py`, `modal_revision.py`, `results/revision/`, `computational_audit/revision/`, and `paper/revision/`. Older root-level paper/results and the legacy `scripts/run_full_extension.py` are retained only for provenance and are superseded. Their custom-wrapper, fine-tuning, gene-network, and earlier numerical results are not part of the revised manuscript.

## Corrected design

- Official scGPT 0.2.5 `scgpt.tasks.embed_data` interface at source commit `cebd6fa`.
- Released whole-human checkpoint pinned by SHA-256 for `args.json`, `vocab.json`, and `best_model.pt`.
- Raw counts with gene filtering and Seurat v3 highly variable gene selection learned from training cells only.
- Identical final genes for the frozen-scGPT and normalized-expression representations within every comparison.
- Two representations crossed with three downstream classifiers, giving 36 primary pipelines across six atlases.
- Five donor-held-out folds across all 199 Indonesia PBMC donors; every preprocessing step is repeated within each outer fold.
- Paired 5,000-draw donor and cell bootstraps.
- Equal-width, adaptive, and classwise ECE, multiclass Brier score, negative log-likelihood, bin counts, and uncertainty intervals.
- A 600-fit label-scarcity analysis from 10 to 500 labelled cells per class.
- Nested-donor temperature scaling evaluated only on untouched outer-test donors.

## Results at a glance

The strongest matched-expression pipeline had higher observed Macro F1 than the strongest frozen-scGPT pipeline in all six within-atlas benchmarks.

| Dataset | Best frozen scGPT | Best matched expression | Difference |
|---|---:|---:|---:|
| TNBC | 0.9799 | 0.9916 | 0.0117 |
| Indonesia PBMC | 0.9357 | 0.9558 | 0.0201 |
| Brain atlas | 0.9314 | 0.9520 | 0.0206 |
| Multi-tissue TME | 0.3653 | 0.4162 | 0.0509 |
| Human pancreas | 0.8736 | 0.9830 | 0.1094 |
| Pig pancreas | 0.8650 | 0.9934 | 0.1284 |

In donor-held-out Indonesia PBMC, pooled Macro F1 was 0.9564 for Matched expression + Logistic Regression and 0.9341 for scGPT embeddings + XGBoost. The paired difference was 0.02236 with a 95% donor-bootstrap interval of 0.02066--0.02423 and was positive in every bootstrap draw.

These results apply to the tested supervised within-atlas and unseen-donor pipelines. They do not establish conclusions about zero-shot mapping, external-atlas transfer, rare-cell discovery, fine-grained states, other checkpoints, or other foundation models.

## Corrected repository map

| Path | Purpose |
|---|---|
| `revision_analysis.py` | Preprocessing, matched representations, classifiers, metrics, bootstraps, scarcity, temperature scaling, aggregation, and plotting. |
| `modal_revision.py` | Pinned cloud runtime, public data downloads, checkpoint audit, GPU validation, analysis stages, and completion assertions. |
| `results/revision/atlas/` | All 36 within-atlas pipelines, paired differences, uncertainty intervals, reliability bins, and aggregate summary. |
| `results/revision/donor/` | All fold and pooled donor-held-out metrics, cell-versus-donor intervals, paired donor bootstraps, and reliability bins. |
| `results/revision/secondary/` | All 600 scarcity fits and all temperature-scaling fold and summary outputs. |
| `computational_audit/revision/` | Dataset audits, matrix/input checks, software environment, source commit, checkpoint hashes, and GPU smoke-test record. |
| `paper/revision/` | Clean line-numbered manuscript, LaTeX sources, main figures, and Supplementary Information. |

## Environment and identities

The reported run used Python 3.11.10, PyTorch 2.5.1 with CUDA 12.4, Scanpy 1.11.5, scikit-learn 1.9.0, XGBoost 3.2.0, and an NVIDIA H100 80 GB GPU. The complete package record is in `computational_audit/revision/software_environment.json`.

Checkpoint SHA-256 values:

```text
args.json       c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60
vocab.json      acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f
best_model.pt   6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922
```

Raw public H5AD files and pretrained weights are intentionally not committed.

## Running the corrected workflow

Install and authenticate the [Modal CLI](https://modal.com/docs/guide), then run the recorded stages from the repository root. For example:

```bash
modal run modal_revision.py --stage prepare-all-datasets
modal run modal_revision.py --stage checkpoint
modal run modal_revision.py --stage smoke
modal run modal_revision.py --stage atlas-benchmarks
modal run modal_revision.py --stage aggregate-atlas-benchmarks
modal run modal_revision.py --stage normalized-donor-folds
modal run modal_revision.py --stage aggregate-normalized-donor-cv
modal run modal_revision.py --stage normalized-donor-secondary-folds
modal run modal_revision.py --stage aggregate-normalized-donor-secondary
```

Every aggregate stage checks completion markers, row counts, test-cell coverage, and alignment before writing final outputs. Exact dataset identifiers and download endpoints are in `modal_revision.py` and the manuscript Data availability section.

## Calibration interpretation

The repository reports probabilities from complete representation--classifier pipelines, not from scGPT intrinsically. Equal-width ECE can favour a different pipeline from Brier score or negative log-likelihood; all measures and reliability-bin counts are retained so the ranking is not reduced to one binned statistic.

## Citation

```bibtex
@article{Khosravi2026AccuracyCalibration,
  title   = {An Accuracy and Calibration Benchmark of scGPT Pipelines and Matched Classical Models for Cell Type Annotation},
  author  = {Khosravi, Alireza and Khosravi, Arshia and {\L}angowski, Kamil and Sieczczy{\'n}ski, Micha{\l} and Pastuszak, Krzysztof and Supernat, Anna and {\.{Z}}aczek, Anna J.},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Revision under review; reproducibility repository}
}
```

## License and contact

Repository code is released under the [MIT License](LICENSE). Public datasets and scGPT weights remain subject to their original licences. For questions, open an issue or contact Anna Supernat at `abednarz@gumed.edu.pl`.
