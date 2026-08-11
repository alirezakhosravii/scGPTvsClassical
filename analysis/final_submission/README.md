# Reproducing and auditing the benchmark

The analysis is split into three auditable stages:

1. `modal_revision_v3.py` downloads the public H5AD assets and audits grouping metadata and source citations.
2. `revision_analysis_v3.py` performs each donor-held-out human fold, the strict PBMC scarcity analysis and the separately designated pig-pancreas stress test.
3. `aggregate_existing_results.py` admits only complete five-fold pipelines, verifies out-of-fold alignment, calculates pooled and per-class metrics, and performs paired donor-cluster bootstrap comparisons. `build_final_artifacts.py` generates the final tables and figures from those compact CSV files.

The GPU functions use the official scGPT 0.2.5 source snapshot at commit `cebd6fa` and the released whole-human checkpoint. The checkpoint SHA-256 hashes are recorded in `../audit/scgpt_checkpoint.json`. Every human result is grouped by `donor_id`; feature selection, SVD fitting and hyperparameter choice occur without access to outer-test donors.

Large H5AD inputs, pretrained weights and prediction matrices are not duplicated in the compact submission archive. The public data identifiers, grouping audit, checkpoint digests and compact aggregate CSVs used for all reported values are included.

## Local, no-GPU reproduction of the reported tables and figures

```bash
python aggregate_existing_results.py --help
python build_final_artifacts.py results/aggregate final_submission_v3
```

The first command requires the completed prediction archives; the second uses only the supplied compact aggregate CSVs. The paper's environment is summarized in `../audit/software_environment.json`.

## Optional full recomputation

`modal_revision_v3.py` and `revision_analysis_v3.py` preserve the full analysis implementation, but rerunning frozen scGPT embedding extraction and model fitting is optional and incurs substantial cloud GPU/CPU cost. It is not required to inspect, compile or reproduce the reported aggregate tables and figures. No paid compute job is required or active for this submission package.
