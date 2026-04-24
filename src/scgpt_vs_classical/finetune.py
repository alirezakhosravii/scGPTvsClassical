"""End-to-end fine-tuning of scGPT with a classification head.

This is a minimal, faithful implementation of the experiment reported in
Table 3 (TNBC). The classification head is a 2-layer MLP on top of the
``<cls>`` token embedding; the encoder is fully unfrozen. We use AdamW with
cosine warm-up/decay, label-smoothing cross-entropy, and report Macro F1
and ECE on the held-out 20% test split.

The defaults below match the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FineTuneConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    label_smoothing: float = 0.05
    head_hidden: int = 256
    dropout: float = 0.2
    grad_accum: int = 1
    seed: int = 42


def build_classification_head(embed_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.2):
    """A 2-layer MLP classification head on top of the <cls> embedding."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(embed_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )


def finetune(
    model,                                    # scgpt model (will be unfrozen)
    binned_expr_train: np.ndarray,
    y_train: np.ndarray,
    binned_expr_test: np.ndarray,
    y_test: np.ndarray,
    gene_ids: np.ndarray,
    cls_token_id: int,
    n_classes: int,
    cfg: FineTuneConfig | None = None,
    device: str = "cuda",
) -> dict:
    """Fine-tune scGPT end-to-end on a single dataset.

    Returns a dict with ``y_pred``, ``y_proba``, ``train_loss``, ``test_loss``.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    cfg = cfg or FineTuneConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Unfreeze encoder
    for p in model.parameters():
        p.requires_grad = True
    model.to(device).train()

    # Build classification head (paper Table 3)
    head = build_classification_head(
        embed_dim=512, n_classes=n_classes,
        hidden=cfg.head_hidden, dropout=cfg.dropout,
    ).to(device)

    g_t = torch.as_tensor(np.concatenate(([cls_token_id], gene_ids)),
                          dtype=torch.long, device=device)

    def _wrap(expr: np.ndarray, y: np.ndarray):
        cls_col = np.zeros((expr.shape[0], 1), dtype=expr.dtype)
        full = np.concatenate([cls_col, expr], axis=1)
        return TensorDataset(
            torch.as_tensor(full, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        )

    train_ds = _wrap(binned_expr_train, y_train)
    test_ds = _wrap(binned_expr_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    optim = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, cfg.epochs * len(train_loader) // max(1, cfg.grad_accum))
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optim, start_factor=1e-2, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    train_losses = []
    for epoch in range(cfg.epochs):
        model.train(); head.train()
        running = 0.0
        optim.zero_grad()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device); y = y.to(device)
            g_batch = g_t.unsqueeze(0).expand(x.size(0), -1)
            enc = model._encode(g_batch, x, src_key_padding_mask=None)
            cls = enc[:, 0, :]
            logits = head(cls)
            loss = loss_fn(logits, y) / cfg.grad_accum
            loss.backward()
            if (step + 1) % cfg.grad_accum == 0:
                optim.step(); sched.step(); optim.zero_grad()
            running += float(loss.item()) * cfg.grad_accum
        train_losses.append(running / max(1, len(train_loader)))

    # Eval
    model.eval(); head.eval()
    all_proba, all_pred, test_loss = [], [], 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device); y = y.to(device)
            g_batch = g_t.unsqueeze(0).expand(x.size(0), -1)
            enc = model._encode(g_batch, x, src_key_padding_mask=None)
            logits = head(enc[:, 0, :])
            test_loss += float(loss_fn(logits, y).item())
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            all_proba.append(proba)
            all_pred.append(proba.argmax(axis=1))
    return {
        "y_pred":  np.concatenate(all_pred),
        "y_proba": np.concatenate(all_proba, axis=0),
        "train_loss": train_losses,
        "test_loss":  test_loss / max(1, len(test_loader)),
    }
