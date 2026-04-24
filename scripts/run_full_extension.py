"""
Full extension script for the scGPT-vs-Classical-ML paper.

Single self-contained script that:
  A. Counts cell types per dataset (fills Table 1 placeholders)
  B. Runs fairness controls (point-4 + point-6) on ALL six datasets
  C. Fine-tunes scGPT end-to-end on TNBC and reports test-set performance
  D. Computes per-class confusion matrices for TNBC and Pig Pancreas
  E. Generates reliability diagrams for every (dataset, model) pair
  F. Bootstraps 95 percent confidence intervals on Macro F1 for every result

Designed to be pasted into a Jupyter notebook on RunPod, cell-by-cell.
The /workspace/model layout is assumed to be the same as the original
benchmark; the script will download datasets and the scGPT checkpoint if
they are not already present.

Outputs:
    /workspace/model/results/Cell_Type_Counts.json
    /workspace/model/results/Fairness_Controls_AllDatasets.csv
    /workspace/model/results/FineTuned_scGPT_TNBC.csv
    /workspace/model/results/Bootstrap_CIs.csv
    /workspace/model/results/predictions/<dataset>__<model>.npz
    /workspace/model/plots/Reliability_<dataset>.pdf
    /workspace/model/plots/ConfusionMatrix_<dataset>.pdf
"""

# =============================================================================
# CELL 1 --- Environment setup and torchtext bypass
# =============================================================================
import os, sys, types, json, gc, urllib.request, warnings, math, time
from pathlib import Path

class DummyVocab:
    def __init__(self, *a, **k): pass
dummy_tt = types.ModuleType("torchtext")
dummy_tt_vocab = types.ModuleType("torchtext.vocab")
dummy_tt_vocab.Vocab = DummyVocab
sys.modules["torchtext"] = dummy_tt
sys.modules["torchtext.vocab"] = dummy_tt_vocab

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from xgboost import XGBClassifier

import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.utils import set_seed

warnings.filterwarnings("ignore")
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# =============================================================================
# CELL 2 --- Paths, dataset URLs, model URL
# =============================================================================
BASE_DIR     = "/workspace/model"
DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
PRED_DIR     = os.path.join(RESULTS_DIR, "predictions")
PLOTS_DIR    = os.path.join(BASE_DIR, "plots")
EMB_DIR      = os.path.join(BASE_DIR, "embeddings")
MODEL_DIR    = BASE_DIR
for d in [DATA_DIR, RESULTS_DIR, PRED_DIR, PLOTS_DIR, EMB_DIR]:
    os.makedirs(d, exist_ok=True)

datasets = {
    "TNBC_Breast_Cancer":  "https://datasets.cellxgene.cziscience.com/af8c4fce-4c63-4671-b339-91a383cf36f6.h5ad",
    "Indonesia_PBMC":      "https://datasets.cellxgene.cziscience.com/665714af-4be5-49a3-913b-5ab5ac25620d.h5ad",
    "Brain_Atlas":         "https://datasets.cellxgene.cziscience.com/0ab54d91-066c-4223-a9ea-6a3b0d1adef4.h5ad",
    "Multi_Tissue_TME":    "https://datasets.cellxgene.cziscience.com/921d46a3-69b4-44a8-b2d6-9ef5c7803bc3.h5ad",
    "Human_Pancreas":      "https://datasets.cellxgene.cziscience.com/00d88707-e33a-4c2a-821a-cdc32a98d050.h5ad",
    "Pig_Pancreas":        "https://datasets.cellxgene.cziscience.com/55cfae87-6348-44df-a4ed-c132569dea54.h5ad",
}

# scGPT human checkpoint (Google Drive folder containing args.json, vocab.json, best_model.pt)
SCGPT_GDRIVE = "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"


# =============================================================================
# CELL 3 --- Download scGPT checkpoint if missing
# =============================================================================
def ensure_scgpt_checkpoint():
    needed = ["args.json", "vocab.json", "best_model.pt"]
    if all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in needed):
        print("scGPT checkpoint already present.")
        return
    print("Downloading scGPT checkpoint via gdown ...")
    os.system("pip install -q gdown")
    os.system(f"cd {MODEL_DIR} && gdown --fuzzy {SCGPT_GDRIVE} -O ./ --folder")
    missing = [f for f in needed if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Still missing after gdown: {missing}. "
            f"Please download manually from {SCGPT_GDRIVE} and place in {MODEL_DIR}."
        )

ensure_scgpt_checkpoint()


# =============================================================================
# CELL 4 --- Helpers (ECE, vocab, model loader, embedding extractor)
# =============================================================================
def calculate_ece(y_true, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies  = predictions == y_true
    ece = 0.0
    bb = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidences > bb[i]) & (confidences <= bb[i+1])
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(confidences[in_bin]) - np.mean(accuracies[in_bin])) * prop
    return ece


class CustomVocab:
    def __init__(self, vocab_dict): self.d = vocab_dict
    def __contains__(self, k): return k in self.d
    def __getitem__(self, k):  return self.d[k]
    def __len__(self):         return len(self.d)
    def append_token(self, t):
        if t not in self.d: self.d[t] = len(self.d)
    def get_stoi(self): return self.d
    def get(self, k, default=None): return self.d.get(k, default)


def load_scgpt_model():
    md = Path(MODEL_DIR)
    with open(md / "vocab.json") as f: vocab = CustomVocab(json.load(f))
    for s in ["<pad>", "<cls>", "<eoc>"]: vocab.append_token(s)
    with open(md / "args.json") as f: configs = json.load(f)
    model = TransformerModel(
        ntoken=len(vocab), d_model=configs.get("embsize", 512),
        nhead=configs.get("nheads", 8), d_hid=configs.get("d_hid", 512),
        nlayers=configs.get("nlayers", 12), vocab=vocab,
        pad_value=-2, n_input_bins=51,
    )
    model.load_state_dict(torch.load(md / "best_model.pt", map_location=device), strict=False)
    model.to(device).eval()
    return model, vocab


def prepare_scgpt_inputs(adata, vocab):
    """Returns (X_binned, gene_ids, kept_genes) ready to feed to the encoder.
    Binning + gene-id remapping for ALL cells, no batching."""
    a = adata.copy()
    sc.pp.filter_genes(a, min_counts=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=1200, flavor="seurat_v3")
    a = a[:, a.var.highly_variable].copy()

    X = a.X.toarray() if hasattr(a.X, "toarray") else a.X
    X_binned = np.digitize(X, np.linspace(0, X.max(), 51)) - 1

    genes = a.var["feature_name"].tolist() if "feature_name" in a.var.columns else a.var_names.tolist()
    gene_ids = np.array([vocab[g] if g in vocab else vocab.get(g.upper(), vocab["<pad>"]) for g in genes])
    valid_mask = gene_ids != vocab["<pad>"]
    kept_genes = [g for g, v in zip(genes, valid_mask) if v]

    gene_ids = gene_ids[valid_mask]
    X_binned = X_binned[:, valid_mask]
    gene_ids = np.insert(gene_ids, 0, vocab["<cls>"])
    X_binned = np.insert(X_binned, 0, 0, axis=1)
    return X_binned.astype(np.int64), gene_ids.astype(np.int64), kept_genes


def extract_embeddings(model, vocab, X_binned, gene_ids, batch_size=1024):
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, X_binned.shape[0], batch_size), desc="encode", leave=False):
            n = X_binned[i:i+batch_size].shape[0]
            src = torch.tensor(gene_ids).unsqueeze(0).repeat(n, 1).to(device)
            val = torch.tensor(X_binned[i:i+batch_size], dtype=torch.float32).to(device)
            try:
                emb = model._encode(src, val, src == vocab["<pad>"])
            except AttributeError:
                emb = model(src, val, src == vocab["<pad>"])["cell_emb"]
            out.append(emb[:, 0, :].cpu().numpy())
    return np.vstack(out)


# =============================================================================
# CELL 5 --- Section A: Cell type counts per dataset
# =============================================================================
def section_A_cell_type_counts():
    print("\n" + "="*70 + "\nSECTION A: Cell-type counts per dataset\n" + "="*70)
    counts_out = {}
    for name, url in datasets.items():
        fp = os.path.join(DATA_DIR, f"{name}.h5ad")
        if not os.path.exists(fp):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, fp)
        a = sc.read_h5ad(fp)
        n_cells_total = a.n_obs
        c = a.obs["cell_type"].value_counts()
        n_types_total = len(c)
        a = a[a.obs["cell_type"].isin(c[c > 5].index)].copy()
        n_cells_kept = a.n_obs
        n_types_kept = a.obs["cell_type"].nunique()
        counts_out[name] = {
            "n_cells_total": int(n_cells_total),
            "n_cells_after_filter": int(n_cells_kept),
            "n_celltypes_total": int(n_types_total),
            "n_celltypes_after_filter": int(n_types_kept),
        }
        print(f"  {name:<22s} cells={n_cells_kept:>7d}   cell_types={n_types_kept}")
        del a; gc.collect()

    with open(os.path.join(RESULTS_DIR, "Cell_Type_Counts.json"), "w") as f:
        json.dump(counts_out, f, indent=2)
    print(f"\nSaved -> {os.path.join(RESULTS_DIR, 'Cell_Type_Counts.json')}")
    return counts_out


# =============================================================================
# CELL 6 --- Section B: Fairness controls on ALL six datasets
#                       Saves probabilities for later reliability diagrams + CIs
# =============================================================================
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "Fairness_Controls_AllDatasets.csv")

all_results = []
if os.path.exists(CHECKPOINT_FILE):
    all_results = pd.read_csv(CHECKPOINT_FILE).to_dict("records")
    print(f"Loaded {len(all_results)} cached results.")

def is_done(ds, model_name):
    return any(r["Dataset"] == ds and r["Model"] == model_name for r in all_results)

def save_result(ds, model_name, f1, ece, n_classes, n_test, note=""):
    all_results.append({
        "Dataset": ds, "Model": model_name,
        "Macro_F1": f1, "ECE": ece,
        "n_classes": int(n_classes), "n_test": int(n_test),
        "Note": note,
    })
    pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, index=False)

def save_predictions(ds, model_name, y_true, y_prob):
    safe_name = model_name.replace("(", "").replace(")", "").replace(" ", "_").replace("+", "and").replace("/", "_")
    fp = os.path.join(PRED_DIR, f"{ds}__{safe_name}.npz")
    np.savez_compressed(fp, y_true=y_true.astype(np.int64), y_prob=y_prob.astype(np.float32))


def fit_and_record(ds, model_name, clf, X_train, y_train, X_test, y_test, note=""):
    if is_done(ds, model_name):
        print(f"  skip {model_name}")
        return
    print(f"  train {model_name}")
    t0 = time.time()
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    f1 = f1_score(y_test, np.argmax(y_prob, 1), average="macro")
    ece = calculate_ece(y_test, y_prob)
    save_result(ds, model_name, f1, ece, n_classes=y_prob.shape[1], n_test=len(y_test), note=note)
    save_predictions(ds, model_name, y_test, y_prob)
    print(f"    F1={f1:.3f}  ECE={ece:.3f}  ({time.time()-t0:.1f}s)")
    del clf; gc.collect()


def section_B_fairness_all_datasets(scgpt_model, vocab):
    print("\n" + "="*70 + "\nSECTION B: Fairness controls on all six datasets\n" + "="*70)
    for name, url in datasets.items():
        print(f"\n--- {name} ---")
        fp = os.path.join(DATA_DIR, f"{name}.h5ad")
        if not os.path.exists(fp):
            urllib.request.urlretrieve(url, fp)
        adata = sc.read_h5ad(fp)
        c = adata.obs["cell_type"].value_counts()
        adata = adata[adata.obs["cell_type"].isin(c[c > 5].index)].copy()
        y = adata.obs["cell_type"].astype("category").cat.codes.values
        train_idx, test_idx = train_test_split(
            np.arange(adata.n_obs), test_size=0.2, random_state=42, stratify=y
        )
        y_train, y_test = y[train_idx], y[test_idx]

        # ---- scGPT embeddings (cached) ---------------------------------
        emb_path   = os.path.join(EMB_DIR, f"{name}_emb.npy")
        genes_path = os.path.join(EMB_DIR, f"{name}_kept_genes.json")
        if os.path.exists(emb_path) and os.path.exists(genes_path):
            embeddings = np.load(emb_path)
            with open(genes_path) as f: kept_genes = json.load(f)
            print(f"  loaded cached embeddings {embeddings.shape}")
        else:
            X_binned, gene_ids, kept_genes = prepare_scgpt_inputs(adata, vocab)
            embeddings = extract_embeddings(scgpt_model, vocab, X_binned, gene_ids)
            np.save(emb_path, embeddings)
            with open(genes_path, "w") as f: json.dump(kept_genes, f)
            print(f"  extracted embeddings {embeddings.shape}")

        # ---- (1) Re-do scGPT_emb + LR for completeness -----------------
        fit_and_record(name, "scGPT_emb + LogReg",
                       LogisticRegression(max_iter=1000, n_jobs=1),
                       embeddings[train_idx], y_train,
                       embeddings[test_idx],  y_test, note="point-4 baseline")
        # ---- (2) scGPT_emb + XGBoost -----------------------------------
        fit_and_record(name, "scGPT_emb + XGBoost",
                       XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                                     n_jobs=-1, tree_method="hist"),
                       embeddings[train_idx], y_train,
                       embeddings[test_idx],  y_test, note="point-4 control")
        # ---- (3) scGPT_emb + RandomForest ------------------------------
        fit_and_record(name, "scGPT_emb + RandomForest",
                       RandomForestClassifier(n_estimators=100, n_jobs=-1),
                       embeddings[train_idx], y_train,
                       embeddings[test_idx],  y_test, note="point-4 control")

        # ---- Vocab-matched feature matrix for classical baselines ------
        a2 = adata.copy()
        sc.pp.filter_genes(a2, min_counts=3)
        sc.pp.normalize_total(a2, target_sum=1e4)
        sc.pp.log1p(a2)
        if "feature_name" in a2.var.columns:
            gene_col = a2.var["feature_name"].astype(str).values
        else:
            gene_col = a2.var_names.astype(str).values
        kept_set = set(kept_genes)
        sel = np.array([g in kept_set for g in gene_col])
        a2 = a2[:, sel].copy()
        Xm = a2.X.toarray() if hasattr(a2.X, "toarray") else a2.X
        print(f"  vocab-matched feature space: {a2.n_vars} genes")

        fit_and_record(name, "LogReg (vocab-matched)",
                       LogisticRegression(max_iter=1000, n_jobs=1),
                       Xm[train_idx], y_train, Xm[test_idx], y_test, note="point-6 control")
        fit_and_record(name, "RandomForest (vocab-matched)",
                       RandomForestClassifier(n_estimators=100, n_jobs=-1),
                       Xm[train_idx], y_train, Xm[test_idx], y_test, note="point-6 control")
        fit_and_record(name, "XGBoost (vocab-matched)",
                       XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                                     n_jobs=-1, tree_method="hist"),
                       Xm[train_idx], y_train, Xm[test_idx], y_test, note="point-6 control")

        del adata, a2, Xm, embeddings; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


# =============================================================================
# CELL 7 --- Section C: Fine-tune scGPT end-to-end on TNBC
#            Adds a linear classification head on top of [CLS]
# =============================================================================
class FineTunedScGPT(nn.Module):
    def __init__(self, base_model, hidden_dim, n_classes, vocab):
        super().__init__()
        self.base = base_model
        self.vocab = vocab
        self.head = nn.Linear(hidden_dim, n_classes)
    def forward(self, src, val):
        pad_mask = src == self.vocab["<pad>"]
        try:
            h = self.base._encode(src, val, pad_mask)
        except AttributeError:
            h = self.base(src, val, pad_mask)["cell_emb"]
        cls = h[:, 0, :]                  # [batch, d_model]
        return self.head(cls)             # [batch, n_classes]


def section_C_finetune_tnbc(scgpt_model, vocab,
                             dataset_name="TNBC_Breast_Cancer",
                             subsample_train=50_000,
                             epochs=3,
                             lr=1e-5,
                             batch_size=64):
    print("\n" + "="*70 + f"\nSECTION C: Fine-tuning scGPT end-to-end on {dataset_name}\n" + "="*70)
    fp = os.path.join(DATA_DIR, f"{dataset_name}.h5ad")
    if not os.path.exists(fp):
        urllib.request.urlretrieve(datasets[dataset_name], fp)
    adata = sc.read_h5ad(fp)
    c = adata.obs["cell_type"].value_counts()
    adata = adata[adata.obs["cell_type"].isin(c[c > 5].index)].copy()
    y = adata.obs["cell_type"].astype("category").cat.codes.values
    n_classes = int(y.max() + 1)
    train_idx, test_idx = train_test_split(
        np.arange(adata.n_obs), test_size=0.2, random_state=42, stratify=y
    )
    y_train, y_test = y[train_idx], y[test_idx]

    X_binned, gene_ids, _ = prepare_scgpt_inputs(adata, vocab)
    print(f"  total cells={adata.n_obs}  n_classes={n_classes}  seq_len={len(gene_ids)}")

    # subsample training set to keep wall-clock manageable
    rng = np.random.RandomState(42)
    if len(train_idx) > subsample_train:
        sel = rng.choice(len(train_idx), size=subsample_train, replace=False)
        train_idx_sub = train_idx[sel]; y_train_sub = y_train[sel]
    else:
        train_idx_sub = train_idx; y_train_sub = y_train
    print(f"  fine-tune train subset: {len(train_idx_sub)}  test: {len(test_idx)}")

    # tensors
    X_train_t = torch.from_numpy(X_binned[train_idx_sub]).float()
    y_train_t = torch.from_numpy(y_train_sub).long()
    X_test_t  = torch.from_numpy(X_binned[test_idx]).float()
    y_test_t  = torch.from_numpy(y_test).long()
    gene_ids_t = torch.from_numpy(gene_ids).long().to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    # Fresh model copy so we do not destroy frozen embeddings used elsewhere
    base = scgpt_model
    embsize = base.encoder.embedding.weight.shape[1] if hasattr(base.encoder, "embedding") else 512
    model_ft = FineTunedScGPT(base, hidden_dim=embsize, n_classes=n_classes, vocab=vocab).to(device)

    optim = torch.optim.AdamW(model_ft.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model_ft.train()
        epoch_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            xb = xb.to(device); yb = yb.to(device)
            src = gene_ids_t.unsqueeze(0).expand(xb.size(0), -1)
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model_ft(src, xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"  epoch {epoch+1} mean train loss = {epoch_loss:.4f}")

    # Evaluation
    model_ft.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in tqdm(test_loader, desc="eval", leave=False):
            xb = xb.to(device)
            src = gene_ids_t.unsqueeze(0).expand(xb.size(0), -1)
            with torch.cuda.amp.autocast():
                logits = model_ft(src, xb)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)
    y_prob = np.vstack(all_probs)
    f1 = f1_score(y_test, np.argmax(y_prob, 1), average="macro")
    ece = calculate_ece(y_test, y_prob)
    print(f"  Fine-tuned scGPT: Macro F1 = {f1:.3f}   ECE = {ece:.3f}")

    pd.DataFrame([{"Dataset": dataset_name, "Model": "scGPT (fine-tuned end-to-end)",
                   "Macro_F1": f1, "ECE": ece,
                   "n_classes": n_classes, "n_test": len(y_test),
                   "Note": f"epochs={epochs} lr={lr} subsample={len(train_idx_sub)}"}]) \
        .to_csv(os.path.join(RESULTS_DIR, "FineTuned_scGPT_TNBC.csv"), index=False)
    save_predictions(dataset_name, "scGPT_finetuned", y_test, y_prob)
    save_result(dataset_name, "scGPT (fine-tuned end-to-end)", f1, ece,
                n_classes=n_classes, n_test=len(y_test),
                note=f"epochs={epochs} lr={lr} subsample={len(train_idx_sub)}")
    del model_ft, base; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# =============================================================================
# CELL 8 --- Section D: Bootstrap 95% CIs on Macro F1 for every prediction file
# =============================================================================
def section_D_bootstrap_cis(n_boot=1000, seed=42):
    print("\n" + "="*70 + "\nSECTION D: Bootstrap 95% CIs on Macro F1\n" + "="*70)
    rng = np.random.RandomState(seed)
    rows = []
    files = sorted(os.listdir(PRED_DIR))
    for fname in tqdm(files, desc="bootstrap"):
        if not fname.endswith(".npz"): continue
        ds, _, model_name = fname.partition("__")
        model_name = model_name.replace(".npz", "")
        d = np.load(os.path.join(PRED_DIR, fname))
        y_true, y_prob = d["y_true"], d["y_prob"]
        y_pred = np.argmax(y_prob, axis=1)
        n = len(y_true)
        boot_f1 = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            boot_f1[b] = f1_score(y_true[idx], y_pred[idx], average="macro")
        f1_mean = float(np.mean(boot_f1))
        f1_lo, f1_hi = float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))
        rows.append({"Dataset": ds, "Model": model_name,
                     "Macro_F1_point": float(f1_score(y_true, y_pred, average="macro")),
                     "Macro_F1_boot_mean": f1_mean,
                     "Macro_F1_CI_low":  f1_lo,
                     "Macro_F1_CI_high": f1_hi,
                     "n_test": int(n), "n_boot": int(n_boot)})
    df = pd.DataFrame(rows).sort_values(["Dataset", "Model"])
    df.to_csv(os.path.join(RESULTS_DIR, "Bootstrap_CIs.csv"), index=False)
    print(f"Saved -> {os.path.join(RESULTS_DIR, 'Bootstrap_CIs.csv')}")
    print(df.to_string(index=False))


# =============================================================================
# CELL 9 --- Section E: Reliability diagrams (one figure per dataset)
# =============================================================================
def reliability_curve(y_true, y_prob, n_bins=10):
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    acc  = pred == y_true
    bb = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_acc, bin_conf, bin_count = [], [], [], []
    for i in range(n_bins):
        m = (conf > bb[i]) & (conf <= bb[i+1])
        if m.sum() > 0:
            bin_centers.append((bb[i] + bb[i+1]) / 2)
            bin_acc.append(np.mean(acc[m]))
            bin_conf.append(np.mean(conf[m]))
            bin_count.append(int(m.sum()))
    return np.array(bin_centers), np.array(bin_acc), np.array(bin_conf), np.array(bin_count)


def section_E_reliability_diagrams():
    print("\n" + "="*70 + "\nSECTION E: Reliability diagrams\n" + "="*70)
    files = [f for f in os.listdir(PRED_DIR) if f.endswith(".npz")]
    by_ds = {}
    for f in files:
        ds = f.split("__")[0]
        by_ds.setdefault(ds, []).append(f)

    for ds, fns in by_ds.items():
        fns = sorted(fns)
        n = len(fns)
        cols = 3; rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows), squeeze=False)
        for ax, fname in zip(axes.flatten(), fns):
            d = np.load(os.path.join(PRED_DIR, fname))
            centers, bin_acc, bin_conf, _ = reliability_curve(d["y_true"], d["y_prob"])
            model_name = fname.split("__", 1)[1].replace(".npz", "").replace("_", " ")
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
            ax.plot(bin_conf, bin_acc, "o-", linewidth=2, markersize=6)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("predicted confidence")
            ax.set_ylabel("observed accuracy")
            ax.set_title(model_name, fontsize=10)
            ax.grid(True, alpha=0.3)
        # hide unused axes
        for ax in axes.flatten()[len(fns):]: ax.axis("off")
        fig.suptitle(f"Reliability diagrams --- {ds}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(PLOTS_DIR, f"Reliability_{ds}.pdf")
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}")


# =============================================================================
# CELL 10 --- Section F: Confusion matrices for TNBC and Pig Pancreas
#             For models: scGPT_emb+LogReg vs LogReg(vocab-matched) vs FT-scGPT
# =============================================================================
def section_F_confusion_matrices(top_k_classes=15):
    print("\n" + "="*70 + "\nSECTION F: Confusion matrices\n" + "="*70)
    target_datasets = ["TNBC_Breast_Cancer", "Pig_Pancreas"]
    target_models = [
        "scGPT_emb_and_LogReg",
        "LogReg_vocab-matched",
        "scGPT_finetuned",
    ]
    for ds in target_datasets:
        # match files defensively (dashes vs underscores can drift)
        avail = [f for f in os.listdir(PRED_DIR) if f.startswith(ds + "__")]
        chosen = []
        for tag in target_models:
            cand = [f for f in avail if tag.replace("-", "_") in f.replace("-", "_")]
            if cand: chosen.append(cand[0])
        if not chosen:
            print(f"  no prediction files for {ds}; skipping"); continue

        # Load actual cell-type labels for axis ticks
        a = sc.read_h5ad(os.path.join(DATA_DIR, f"{ds}.h5ad"))
        c = a.obs["cell_type"].value_counts()
        a = a[a.obs["cell_type"].isin(c[c > 5].index)].copy()
        cat = a.obs["cell_type"].astype("category")
        class_names = list(cat.cat.categories)
        # restrict to top-K most frequent for legibility
        top_classes = c[c > 5].index[:top_k_classes].tolist()
        keep_codes = [class_names.index(cn) for cn in top_classes if cn in class_names]
        keep_codes_set = set(keep_codes)

        n = len(chosen)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5))
        if n == 1: axes = [axes]
        for ax, fname in zip(axes, chosen):
            d = np.load(os.path.join(PRED_DIR, fname))
            y_true = d["y_true"]; y_pred = np.argmax(d["y_prob"], axis=1)
            mask = np.array([y in keep_codes_set for y in y_true])
            cm = confusion_matrix(y_true[mask], y_pred[mask],
                                  labels=keep_codes, normalize="true")
            sns.heatmap(cm, ax=ax, cmap="Blues", vmin=0, vmax=1, cbar=True,
                        xticklabels=top_classes, yticklabels=top_classes)
            model_name = fname.split("__", 1)[1].replace(".npz", "").replace("_", " ")
            ax.set_title(model_name, fontsize=10)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.tick_params(axis="x", rotation=90, labelsize=7)
            ax.tick_params(axis="y", rotation=0, labelsize=7)
        fig.suptitle(f"Per-class confusion matrices --- {ds} (row-normalised; top-{top_k_classes} classes)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(PLOTS_DIR, f"ConfusionMatrix_{ds}.pdf")
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}")


# =============================================================================
# CELL 11 --- Run all sections
# =============================================================================
print("Loading scGPT model ...")
scgpt_model, vocab = load_scgpt_model()
print("scGPT loaded.")

# A. Cell-type counts (cheap, always run)
section_A_cell_type_counts()

# B. Fairness controls on all 6 datasets (the heavy lifting)
section_B_fairness_all_datasets(scgpt_model, vocab)

# C. Fine-tune scGPT end-to-end on TNBC
#    NOTE: requires gradient flow through the encoder, so cannot share the same
#          frozen module instance with section B. We reload a fresh copy here.
print("\nReloading a fresh scGPT model for fine-tuning ...")
ft_model, ft_vocab = load_scgpt_model()
section_C_finetune_tnbc(ft_model, ft_vocab,
                        dataset_name="TNBC_Breast_Cancer",
                        subsample_train=50_000, epochs=3, lr=1e-5, batch_size=64)
del ft_model; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# D. Bootstrap 95% CIs on Macro F1 (uses saved predictions from B and C)
section_D_bootstrap_cis(n_boot=1000)

# E. Reliability diagrams (one figure per dataset; uses saved probabilities)
section_E_reliability_diagrams()

# F. Confusion matrices (TNBC, Pig Pancreas)
section_F_confusion_matrices(top_k_classes=15)

print("\n" + "="*70 + "\nALL SECTIONS COMPLETE\n" + "="*70)
print(f"Results CSV   : {CHECKPOINT_FILE}")
print(f"Cell counts   : {os.path.join(RESULTS_DIR, 'Cell_Type_Counts.json')}")
print(f"Fine-tune CSV : {os.path.join(RESULTS_DIR, 'FineTuned_scGPT_TNBC.csv')}")
print(f"Bootstrap CIs : {os.path.join(RESULTS_DIR, 'Bootstrap_CIs.csv')}")
print(f"Plots         : {PLOTS_DIR}")
print(f"Predictions   : {PRED_DIR}")
