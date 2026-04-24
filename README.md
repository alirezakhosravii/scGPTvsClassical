# scGPTvsClassical

> **Calibration-aware benchmarking of scGPT against classical machine learning for cell-type annotation across six single-cell atlases (~1.3 M cells).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20artefact-success)

This repository contains the full analysis pipeline, intermediate results, figures, and LaTeX source for the paper:

> **Khosravi A, Khosravi A, Łangowski K, Sieczczyński M, Pastuszak K, Supernat A, Żaczek AJ.**
> *Classical Machine Learning Outperforms and Excels in Calibration Compared to Single-Cell Large Language Models for Cell-Type Annotation at Scale.*.

The headline finding is simple: across six diverse atlases (TNBC, Indonesian PBMC, brain, multi-tissue tumour microenvironment, human and pig pancreas), classical Logistic Regression on highly variable genes restricted to scGPT's vocabulary **matches or beats every scGPT-based pipeline we tested** — including frozen embeddings with stronger non-linear probes (XGBoost, Random Forest) and full end-to-end fine-tuning of scGPT itself — in both predictive accuracy (Macro F1) and statistical calibration (Expected Calibration Error).

---

## Table of Contents

- [What's in this repository](#whats-in-this-repository)
- [Quick start](#quick-start)
- [Reproducing the paper end-to-end](#reproducing-the-paper-end-to-end)
- [Repository layout](#repository-layout)
- [Datasets](#datasets)
- [Results at a glance](#results-at-a-glance)
- [How to cite](#how-to-cite)
- [License](#license)

---

## What's in this repository

| Path | Content |
|---|---|
| `src/scgpt_vs_classical/` | A small, well-documented Python library with the calibration metrics, scGPT preprocessing/embedding code, classical baselines, end-to-end fine-tuning loop, and plotting utilities. |
| `scripts/run_full_extension.py` | The single entry-point script that produces every fairness-control number, every bootstrap CI, every reliability diagram, and the end-to-end fine-tuning result reported in the paper. |
| `scripts/00_download_datasets.sh` | One-liner for fetching all six `.h5ad` files from the CZ CELLxGENE Discover portal. |
| `results/` | The exact CSV outputs (`Fairness_Controls_AllDatasets.csv`, `Bootstrap_CIs.csv`) and PDF figures referenced by the paper. |
| `paper/` | LaTeX source, BibTeX file, and the compiled PDF of the manuscript. |
| `notebooks/` | Optional Jupyter notebooks for exploratory analysis (kept minimal on purpose; the canonical entry point is the script). |

Everything that the paper claims — every number in Table 2, every CI in Table 3, every reliability diagram in the Supplement — comes from this code, applied to the public datasets listed below.

---

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/alirezakhosravii/scGPTvsClassical.git
cd scGPTvsClassical

# 2. Create a fresh Python 3.10+ environment (conda shown; venv works too)
conda create -n scgpt-bench python=3.10 -y
conda activate scgpt-bench

# 3. Install dependencies
pip install -r requirements.txt
# scGPT is the trickiest dep — see the note at the bottom of this section.

# 4. Download the six datasets (~5 GB total)
bash scripts/00_download_datasets.sh data/raw

# 5. (Optional) Download the pre-trained scGPT whole-human checkpoint:
#    https://github.com/bowang-lab/scGPT  →  "Pretrained scGPT model zoo"
#    Unpack into  models/scGPT_human/

# 6. Run the full benchmark
python scripts/run_full_extension.py \
       --data-dir   data/raw \
       --model-dir  models/scGPT_human \
       --out-dir    results/
```

> **Note on installing scGPT.** The official `scgpt` package depends on `flash-attn` and a specific `torch`/`torchtext` pair. We have had the most success on Linux + CUDA 12.1 with `torch==2.1.0` and `torchtext==0.16.0`, installed *before* `scgpt`. Mac users will need to fall back to CPU or use a remote GPU.

---

## Reproducing the paper end-to-end

The full pipeline runs in three stages, all driven by `scripts/run_full_extension.py`:

1. **Frozen-representation pipelines.** For each of the six datasets:
   - extract `<cls>` embeddings from frozen scGPT,
   - train Logistic Regression, XGBoost, and Random Forest on those embeddings,
   - train Logistic Regression, XGBoost, and Random Forest on the **same** highly variable genes restricted to scGPT's 60,697-token vocabulary (the "vocab-matched" controls),
   - report Macro F1 with bootstrap 95% CIs and ECE for all six classifiers.

2. **End-to-end fine-tuning** (TNBC only, mirroring paper Table 3):
   - unfreeze the scGPT encoder,
   - attach a 2-layer MLP classification head,
   - train with AdamW + cosine schedule + label smoothing for 8 epochs,
   - report Macro F1 (with CI) and ECE on the held-out 20% test split.

3. **Figures.**
   - 6-panel reliability diagrams per dataset (`results/figures/Reliability_*.pdf`),
   - per-class confusion matrices for TNBC and Pig Pancreas (`results/figures/ConfusionMatrix_*.pdf`).

On a single NVIDIA H100 80 GB the full run takes ≈ 6 hours. The dominant cost is the embedding extraction for the two largest atlases (TNBC, PBMC); everything else fits in minutes. CPU-only execution is technically possible for the classical baselines but not for scGPT.

A successful run regenerates exactly the two CSVs in `results/` and exactly the eight PDFs in `results/figures/`.

---

## Repository layout

```
scGPTvsClassical/
├── README.md              ← you are here
├── LICENSE                ← MIT
├── CITATION.cff           ← machine-readable citation metadata
├── requirements.txt       ← Python dependencies
├── .gitignore
│
├── src/
│   └── scgpt_vs_classical/
│       ├── __init__.py
│       ├── calibration.py     ← ECE, reliability curves, bootstrap CIs
│       ├── scgpt_pipeline.py  ← HVG selection, vocab matching, embedding extraction
│       ├── classical.py       ← XGBoost / RF / LogReg trainers
│       ├── finetune.py        ← end-to-end scGPT fine-tuning (paper Table 3)
│       └── plotting.py        ← reliability grids and confusion matrices
│
├── scripts/
│   ├── 00_download_datasets.sh
│   └── run_full_extension.py  ← single entry point reproducing the paper
│
├── results/
│   ├── Fairness_Controls_AllDatasets.csv  ← Table 2 in the paper
│   ├── Bootstrap_CIs.csv                  ← every bootstrap CI in the paper
│   └── figures/
│       ├── Reliability_TNBC_Breast_Cancer.pdf
│       ├── Reliability_Indonesia_PBMC.pdf
│       ├── Reliability_Brain_Atlas.pdf
│       ├── Reliability_Multi_Tissue_TME.pdf
│       ├── Reliability_Human_Pancreas.pdf
│       ├── Reliability_Pig_Pancreas.pdf
│       ├── ConfusionMatrix_TNBC_Breast_Cancer.pdf
│       └── ConfusionMatrix_Pig_Pancreas.pdf
│
├── paper/
│   ├── main.tex            ← BMC Bioinformatics manuscript
│   ├── references.bib
│   └── main.pdf            ← compiled PDF (28 pages)
│
└── notebooks/              ← optional exploratory notebooks
```

---

## Datasets

All six datasets are public and distributed by the Chan Zuckerberg CELLxGENE Discover portal (<https://cellxgene.cziscience.com/>). The exact files we used:

| Dataset | Cells | Cell types | Species | Direct URL |
|---|---:|---:|---|---|
| TNBC Breast Cancer | 427,823 | 7  | *H. sapiens* | [`af8c4fce-…`](https://datasets.cellxgene.cziscience.com/af8c4fce-4c63-4671-b339-91a383cf36f6.h5ad) |
| Indonesia PBMC     | 462,034 | 13 | *H. sapiens* | [`665714af-…`](https://datasets.cellxgene.cziscience.com/665714af-4be5-49a3-913b-5ab5ac25620d.h5ad) |
| Brain Atlas        |  75,583 | 10 | *H. sapiens* | [`0ab54d91-…`](https://datasets.cellxgene.cziscience.com/0ab54d91-066c-4223-a9ea-6a3b0d1adef4.h5ad) |
| Multi-Tissue TME   | 391,963 | 11 | *H. sapiens* | [`921d46a3-…`](https://datasets.cellxgene.cziscience.com/921d46a3-69b4-44a8-b2d6-9ef5c7803bc3.h5ad) |
| Human Pancreas     |  26,474 |  4 | *H. sapiens* | [`00d88707-…`](https://datasets.cellxgene.cziscience.com/00d88707-e33a-4c2a-821a-cdc32a98d050.h5ad) |
| Pig Pancreas       |  22,056 |  4 | *S. scrofa*  | [`55cfae87-…`](https://datasets.cellxgene.cziscience.com/55cfae87-6348-44df-a4ed-c132569dea54.h5ad) |

The scGPT pre-trained whole-human model weights (~ 1.4 GB) come from the original Wang Lab repository: <https://github.com/bowang-lab/scGPT>.

---

## Results at a glance

| Dataset            | scGPT + LogReg<br>(Macro F1 / ECE) | scGPT + XGB<br>(Macro F1 / ECE) | LogReg vocab-matched<br>(Macro F1 / ECE) |
|---|---|---|---|
| TNBC Breast Cancer | 0.715 / 0.183 | 0.972 / 0.034 | **0.993 / 0.005** |
| Indonesia PBMC     | 0.273 / 0.123 | 0.617 / 0.022 | **0.953 / 0.007** |
| Brain Atlas        | 0.412 / 0.097 | 0.821 / 0.018 | **0.968 / 0.011** |
| Multi-Tissue TME   | 0.491 / 0.155 | **0.896 / 0.014** | 0.898 / 0.013 |
| Human Pancreas     | 0.683 / 0.071 | 0.954 / 0.009 | **0.991 / 0.004** |
| Pig Pancreas       | 0.331 / 0.118 | 0.748 / 0.026 | **0.946 / 0.008** |

**End-to-end fine-tuning of scGPT on TNBC** reached Macro F1 = 0.975 / ECE = 0.003 — statistically indistinguishable from the much simpler scGPT + XGBoost pipeline (0.972) and still below the vocabulary-matched LogReg baseline (0.993).

See `paper/main.pdf` Table 2 for the full 6-dataset × 6-model fairness-control matrix with bootstrap 95% CIs, and Table 3 for the head-to-head fine-tuning comparison.

---

## How to cite

If you use this code or the released CSVs / figures, please cite:

```bibtex
@article{Khosravi2026scGPTvsClassical,
  title   = {Classical Machine Learning Outperforms and Excels in Calibration
             Compared to Single-Cell Large Language Models for Cell-Type
             Annotation at Scale},
  author  = {Khosravi, Alireza and Khosravi, Arshia and {\L}angowski, Kamil
             and Sieczczy{\'n}ski, Micha{\l} and Pastuszak, Krzysztof
             and Supernat, Anna and {\.Z}aczek, Anna J.},
  journal = {BMC Bioinformatics},
  year    = {2026},
  note    = {Under review. Code: \url{https://github.com/alirezakhosravii/scGPTvsClassical}}
}
```

A machine-readable `CITATION.cff` is also provided.

---

## License

This repository is released under the [MIT License](LICENSE). The pre-trained scGPT weights are distributed under their original licence by the Wang Lab; please consult the [scGPT repository](https://github.com/bowang-lab/scGPT) for those terms. The single-cell datasets are distributed by their respective original publishers under the terms set by the CZ CELLxGENE Discover portal.

---

## Contact

For questions about the paper or the code, please open a GitHub issue, or contact the corresponding author **Anna Supernat** (`abednarz@gumed.edu.pl`).
