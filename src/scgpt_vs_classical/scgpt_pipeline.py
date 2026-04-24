"""scGPT preprocessing, vocabulary matching and frozen embedding extraction.

Mirrors the protocol described in the paper Methods section:
    * Seurat v3 HVG selection on raw counts (top 1,200 genes).
    * Binning of expression into 51 discrete bins.
    * Mapping to scGPT's 60,697-token vocabulary; out-of-vocab genes are
      replaced by ``<pad>``.
    * Prepending a ``<cls>`` token and extracting its 512-dimensional
      embedding from the frozen encoder.

This module deliberately exposes the per-step functions instead of one giant
end-to-end function, so users can swap in their own dataset loader, reuse the
vocabulary mapping for a different downstream learner, or cache embeddings.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


N_HVG_DEFAULT = 1_200
N_BINS_DEFAULT = 51
EMBED_DIM = 512


def select_hvgs_seurat_v3(adata, n_top_genes: int = N_HVG_DEFAULT):
    """Run Scanpy's Seurat v3 HVG selection on raw counts in-place."""
    import scanpy as sc

    adata = adata.copy()
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        subset=True,
    )
    return adata


def bin_expression(X: np.ndarray, n_bins: int = N_BINS_DEFAULT) -> np.ndarray:
    """Bin a (n_cells, n_genes) expression matrix into ``n_bins`` integer bins.

    Each row is binned independently across ``[0, row.max()]``.
    Zeros stay in bin 0.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_cells, n_genes).")
    out = np.zeros_like(X, dtype=np.int64)
    for i in range(X.shape[0]):
        row = X[i]
        m = float(row.max())
        if m <= 0:
            continue
        edges = np.linspace(0.0, m, n_bins + 1)
        out[i] = np.digitize(row, edges[1:-1])
    return out


def match_to_vocab(
    gene_names: Iterable[str],
    vocab,                           # scgpt.tokenizer.gene_tokenizer.GeneVocab
    pad_token: str = "<pad>",
) -> tuple[np.ndarray, float]:
    """Map gene symbols to scGPT vocabulary IDs.

    Returns
    -------
    ids : (n_genes,) int array of token IDs (``<pad>`` where unmatched).
    coverage : fraction of HVGs successfully matched in [0, 1].
    """
    pad_id = vocab[pad_token]
    matched = 0
    ids = np.empty(len(list(gene_names)) if hasattr(gene_names, "__len__") else 0,
                   dtype=np.int64)
    out = []
    for g in gene_names:
        if g in vocab:
            out.append(vocab[g])
            matched += 1
        else:
            out.append(pad_id)
    ids = np.asarray(out, dtype=np.int64)
    coverage = matched / max(1, len(ids))
    return ids, coverage


def extract_cls_embeddings(
    model,                           # scgpt model in eval mode
    binned_expr: np.ndarray,         # (n_cells, n_genes) int
    gene_ids: np.ndarray,            # (n_genes,) int, vocab IDs
    cls_token_id: int,
    batch_size: int = 1024,
    device: str = "cuda",
) -> np.ndarray:
    """Pass cells through the frozen scGPT encoder and return the <cls> token.

    Returns
    -------
    embeddings : (n_cells, 512) float32 array.
    """
    import torch

    n_cells = binned_expr.shape[0]
    out = np.empty((n_cells, EMBED_DIM), dtype=np.float32)
    model.eval()

    # Prepend <cls> column to gene IDs and to the binned expression rows.
    gene_ids_full = np.concatenate(([cls_token_id], gene_ids))
    cls_expr_col = np.zeros((n_cells, 1), dtype=binned_expr.dtype)
    expr_full = np.concatenate([cls_expr_col, binned_expr], axis=1)

    g_t = torch.as_tensor(gene_ids_full, dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            x = torch.as_tensor(expr_full[start:end], dtype=torch.float32, device=device)
            g_batch = g_t.unsqueeze(0).expand(end - start, -1)
            # _encode returns (batch, seq, embed); take the <cls> token at idx 0
            enc = model._encode(g_batch, x, src_key_padding_mask=None)
            out[start:end] = enc[:, 0, :].cpu().numpy()
    return out
