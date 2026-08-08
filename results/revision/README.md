# Corrected revision results

These files are the aggregate numerical record for the Scientific Reports revision titled **An Accuracy and Calibration Benchmark of scGPT Pipelines and Matched Classical Models for Cell Type Annotation**.

- `atlas/`: all 36 within-atlas pipelines, cell-bootstrap intervals, paired differences, reliability bins, and aggregate summary.
- `donor/`: five-fold and pooled Indonesia PBMC results, donor-versus-cell intervals, paired donor bootstraps, and donor-held-out reliability bins.
- `secondary/`: every label-scarcity fit and fold summary plus every temperature-scaling fold and summary.

Model labels containing `Matched HVGs` refer to library-size-normalized, `log1p`-transformed expression of the exact training-selected genes matched to the scGPT vocabulary. The manuscript uses the shorter display label `Matched expression`.

Raw per-cell prediction archives are omitted because of size. Aggregate code asserts exact cell coverage, class alignment, and paired test indices before writing these files.
