# scGPTvsClassical

> Corrected reproducible benchmark of frozen official scGPT embeddings versus matched normalized expression baselines for cell type annotation in single-cell atlases.

This repository contains the corrected analysis workflow, aggregate results, figures, and LaTeX materials for the revised manuscript:

**Accuracy and Calibration of scGPT Pipelines and Classical Models for Cell Type Annotation in Single Cell Atlases**

## Revision status

This revision supersedes earlier analysis workflows that used a custom embedding wrapper and sparse XGBoost handling. The corrected pipeline uses:

- official scGPT 0.2.5 `scgpt.tasks.embed_data` implementation
- scGPT commit `cebd6fa`
- training-only highly variable gene selection
- vocabulary-matched genes for classical expression baselines
- dense XGBoost input
- five donor-held-out folds covering 199 donors
- 5,000 bootstrap resamples
- 600 label-scarcity fits
- five calibration measures
- held-out-donor temperature scaling

## Benchmark design

All comparisons evaluate complete pipelines:

1. Frozen scGPT embeddings coupled to Logistic Regression, Random Forest, or XGBoost.
2. Matched normalized expression coupled to the same classical classifiers.

The main conclusion is restricted to the tested within-atlas supervised annotation setting. Results do not imply conclusions about zero-shot annotation, external atlas transfer, rare or fine-grained labels, multimodal tasks, or foundation models generally.

## Corrected headline results

The donor-held-out benchmark included 1,405,933 source cells. Matched expression pipelines achieved higher donor-held-out Macro F1 than scGPT embedding pipelines:

- matched expression: Macro F1 = 0.9564
- scGPT embeddings: Macro F1 = 0.9341
- paired donor bootstrap difference: 0.02236
- 95% donor bootstrap interval: 0.02066–0.02423

Label scarcity experiments and calibration analyses are provided in the aggregate results directory.

## Reproducibility

The repository contains:

- corrected aggregate CSV outputs
- manuscript LaTeX sources
- replacement figures
- environment specifications
- analysis metadata and audit records

Previous workflows are retained only as historical records and are marked as superseded.

## Contact

For manuscript correspondence, see the published manuscript metadata and submission files.
