#!/usr/bin/env python3
"""
test_jepa_embeddings.py — Unified JEPA evaluation suite
Section 1: Embedding sanity (collapse, PCA rank, t-SNE)
Section 2: Linear probe (next-day return direction prediction)
Section 3: Unified verdict
"""

import os
import sys
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.insert(0, '.')
from src.models.encoder import Encoder
from individual_stocks.data_class_parquet import StockMarketJEPADataset
from individual_stocks.dataextraction import DataExtractor

# --- Config ----------------------------------------------------------------
MODEL_PATH    = "jepa-model/model_3_epoch_50.pt"
TRAIN_PARQUET = "individual_stocks/parquet_data/sector_etf_clean_trainingset.parquet"
TEST_PARQUET  = "individual_stocks/parquet_data/sector_etf_clean_testingset.parquet"
OUTPUT_DIR    = "jepa-model/analysis/iteration-3"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_TICKER  = "XLK"
NUM_PROBE_EPOCHS = 50

ENC_DIM_IN        = 49
ENC_NUM_PATCHES   = 20
ENC_KERNEL_SIZE   = 49
ENC_EMBED_DIM     = 64
ENC_NHEAD         = 8
ENC_NUM_LAYERS    = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# LOAD MODEL (shared)
# ============================================================================
print("=" * 60)
print("JEPA Embedding Evaluation Suite")
print("=" * 60)

encoder = Encoder(
    dim_in=ENC_DIM_IN,
    num_patches=ENC_NUM_PATCHES,
    kernel_size=ENC_KERNEL_SIZE,
    embed_dim=ENC_EMBED_DIM,
    embed_bias=True,
    nhead=ENC_NHEAD,
    jepa=True,
    num_layers=ENC_NUM_LAYERS,
)
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
encoder.load_state_dict(state)
encoder.to(DEVICE)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False
print(f"Loaded encoder from {MODEL_PATH}")
print(f"Total params: {sum(p.numel() for p in encoder.parameters()):,}")


# ============================================================================
# SECTION 1: EMBEDDING SANITY
# ============================================================================
print(f"\n{'═' * 60}")
print(f"SECTION 1: Embedding Sanity Check")
print(f"{'═' * 60}")

dataset = StockMarketJEPADataset(
    mask_ratio=0.2, num_patches=20, vix_fairweather=20,
    parquet_path=TRAIN_PARQUET,
)
test_dataset = StockMarketJEPADataset(
    mask_ratio=0.2, num_patches=20, vix_fairweather=20,
    parquet_path=TEST_PARQUET,
)
print(f"Train windows (non-overlapping): {len(dataset)}")
print(f"Test windows (non-overlapping):  {len(test_dataset)}")


@torch.no_grad()
def get_embeddings(ds, max_windows=None):
    windows = []
    n = len(ds) if max_windows is None else min(len(ds), max_windows)
    for i in range(n):
        window, _, _ = ds[i]
        windows.append(window.unsqueeze(0))
    x = torch.cat(windows, dim=0).to(DEVICE)
    emb = encoder(x)
    pooled = emb.mean(dim=1)
    return pooled.cpu().numpy()

train_emb = get_embeddings(dataset, max_windows=500)
test_emb  = get_embeddings(test_dataset, max_windows=500)
print(f"Extracted {train_emb.shape[0]} train embeddings, {test_emb.shape[0]} test embeddings (dim={train_emb.shape[1]})")


# -- 1a. Pairwise cosine similarity -----------------------------------------
def cosine_similarity_matrix(emb):
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb_norm @ emb_norm.T

train_sim = cosine_similarity_matrix(train_emb)
test_sim  = cosine_similarity_matrix(test_emb)

triu_idx = np.triu_indices_from(train_sim, k=1)
triu_idx_test = np.triu_indices_from(test_sim, k=1)
mean_train_sim = train_sim[triu_idx].mean()
std_train_sim  = train_sim[triu_idx].std()
mean_test_sim  = test_sim[triu_idx_test].mean()
std_test_sim   = test_sim[triu_idx_test].std()

print(f"\n{'─' * 60}")
print(f"Pairwise Cosine Similarity")
print(f"{'─' * 60}")
print(f"   Train — mean: {mean_train_sim:.4f}  +/- {std_train_sim:.4f}")
print(f"   Test  — mean: {mean_test_sim:.4f}  +/- {std_test_sim:.4f}")

if mean_train_sim > 0.95:
    collapse_msg = "WARNING: >0.95 similarity -- likely representation collapse"
elif mean_train_sim > 0.85:
    collapse_msg = "CAUTION: >0.85 similarity -- embeddings are quite similar"
else:
    collapse_msg = "OK: embeddings are diverse"

print(f"   Verdict: {collapse_msg}")


# -- 1b. PCA effective rank -------------------------------------------------
pca = PCA(n_components=min(50, train_emb.shape[1]))
pca.fit(train_emb)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_90 = np.searchsorted(cum_var, 0.90) + 1
n_95 = np.searchsorted(cum_var, 0.95) + 1

print(f"\nPCA on train embeddings:")
print(f"   Dims for 90% variance: {n_90} / {train_emb.shape[1]}")
print(f"   Dims for 95% variance: {n_95} / {train_emb.shape[1]}")

if n_90 <= 3:
    pca_msg = "WARNING: 90% variance in <=3 dims -- likely rank collapse"
else:
    pca_msg = "OK: >3 effective dimensions with real structure"
print(f"   Verdict: {pca_msg}")


# -- 1c. Train/Test distribution shift --------------------------------------
train_mean = train_emb.mean(axis=0)
test_mean  = test_emb.mean(axis=0)
mean_shift = np.linalg.norm(train_mean - test_mean)

pca_test = PCA(n_components=min(50, test_emb.shape[1]))
pca_test.fit(test_emb)
test_90 = np.searchsorted(np.cumsum(pca_test.explained_variance_ratio_), 0.90) + 1

print(f"\nTrain/Test distribution shift:")
print(f"   Mean embedding shift (L2): {mean_shift:.4f}")
print(f"   Test  dims for 90% var:    {test_90} / {test_emb.shape[1]}")
print(f"   Train dims for 90% var:    {n_90} / {train_emb.shape[1]}")


# -- 1d. t-SNE visualization ------------------------------------------------
n_tsne = min(300, len(train_emb))
indices = np.random.RandomState(42).choice(len(train_emb), n_tsne, replace=False)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_2d = tsne.fit_transform(train_emb[indices])

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=indices, cmap='viridis', s=15, alpha=0.7)
ax.set_title("t-SNE of JEPA Embeddings (Train Set)")
ax.set_xlabel("t-SNE dim 1")
ax.set_ylabel("t-SNE dim 2")
plt.colorbar(sc, label="Window index")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tsne_embeddings.png", dpi=150)
plt.close()
print(f"t-SNE plot saved to {OUTPUT_DIR}/tsne_embeddings.png")


# ============================================================================
# SECTION 2: LINEAR PROBE
# ============================================================================
print(f"\n{'═' * 60}")
print(f"SECTION 2: Linear Probe (next-day {PROBE_TICKER} direction)")
print(f"{'═' * 60}")

extractor = DataExtractor(ticker_list=None, macro_indices=None, dataset=TRAIN_PARQUET)
etf_lr, etf_sp, etf_rvol = extractor.get_assets()
macro_lr, macro_sp, macro_rvol = extractor.get_macro()
vix = extractor.get_vix(vix_fairweather=20)

vix_aligned = pd.Series(vix.values, index=etf_lr.index, name='VIX')

all_features = pd.concat([
    etf_lr, etf_sp, etf_rvol,
    macro_lr, macro_sp, macro_rvol,
    vix_aligned
], axis=1).dropna()
data_np = all_features.values.astype(np.float32)
print(f"Full feature matrix: {data_np.shape} (days x 49)")

xlk_returns_aligned = etf_lr.loc[all_features.index, PROBE_TICKER].values.astype(np.float32)
xlk_col = etf_lr.columns.get_loc(PROBE_TICKER)

windows_49, labels_sliding = [], []
for i in range(len(data_np) - ENC_NUM_PATCHES - 1):
    w = data_np[i:i + ENC_NUM_PATCHES]
    next_ret = xlk_returns_aligned[i + ENC_NUM_PATCHES]
    if not np.isnan(next_ret):
        windows_49.append(w)
        labels_sliding.append(1.0 if next_ret > 0 else 0.0)

print(f"Sliding windows: {len(windows_49)}")
print(f"Label balance: {np.mean(labels_sliding):.3f} positive")

split_idx = int(len(windows_49) * 0.8)
train_w = windows_49[:split_idx]
train_l = labels_sliding[:split_idx]
val_w = windows_49[split_idx:]
val_l = labels_sliding[split_idx:]
print(f"Train: {len(train_w)}, Val: {len(val_w)}")


@torch.no_grad()
def extract_embs(windows_list, batch_size=64):
    all_embs = []
    for i in range(0, len(windows_list), batch_size):
        batch = windows_list[i:i + batch_size]
        x = torch.tensor(np.array(batch), dtype=torch.float32).to(DEVICE)
        emb = encoder(x)
        pooled = emb.mean(dim=1)
        all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)

print("Extracting train embeddings...")
train_emb_sliding = extract_embs(train_w)
print("Extracting val embeddings...")
val_emb_sliding = extract_embs(val_w)

mean_emb = train_emb_sliding.mean(dim=0, keepdim=True)
train_emb_c = train_emb_sliding - mean_emb
val_emb_c = val_emb_sliding - mean_emb

train_labels = torch.tensor(train_l, dtype=torch.float32)
val_labels = torch.tensor(val_l, dtype=torch.float32)


# -- 2a. Train linear probe -------------------------------------------------
probe = torch.nn.Linear(ENC_EMBED_DIM, 1).to(DEVICE)
optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
bce = torch.nn.BCEWithLogitsLoss()

probe_train_losses, probe_val_losses = [], []
probe_train_accs, probe_val_accs = [], []

best_val_acc = 0.0
for epoch in range(NUM_PROBE_EPOCHS):
    probe.train()
    perm = torch.randperm(len(train_emb_c))
    total_loss = 0.0
    correct = 0
    for i in range(0, len(perm), 32):
        idx = perm[i:i + 32]
        x = train_emb_c[idx].to(DEVICE)
        y = train_labels[idx].to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        logits = probe(x)
        loss = bce(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = (logits > 0).float()
        correct += (preds == y).sum().item()

    train_acc = correct / len(train_emb_c)

    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_emb_c.to(DEVICE))
        val_loss = bce(val_logits, val_labels.to(DEVICE).unsqueeze(1))
        val_preds = (val_logits > 0).float()
        val_acc = (val_preds.squeeze() == val_labels.to(DEVICE)).float().mean().item()

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    probe_train_losses.append(total_loss / (len(train_emb_c) // 32 + 1))
    probe_val_losses.append(val_loss.item())
    probe_train_accs.append(train_acc)
    probe_val_accs.append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:2d} | train acc: {train_acc:.3f} | val acc: {val_acc:.3f} | best val: {best_val_acc:.3f}")


# -- 2b. Majority-class baseline -------------------------------------------
majority_is_down = train_l.count(0) > train_l.count(1)
baseline_acc = 1 - np.mean(train_l) if majority_is_down else np.mean(train_l)

print(f"\nMajority-class baseline acc: {baseline_acc:.3f}")
print(f"Linear probe best val acc:    {best_val_acc:.3f}")

if best_val_acc > baseline_acc + 0.03:
    probe_msg = "SIGNAL: embeddings contain real market information"
elif best_val_acc > baseline_acc:
    probe_msg = "WEAK: slightly better than majority baseline"
else:
    probe_msg = "NO SIGNAL: not better than majority baseline"
print(f"Verdict: {probe_msg}")


# -- 2c. Raw XLK returns sanity probe ---------------------------------------
print(f"\nSanity check: linear probe on raw {PROBE_TICKER} returns (20d window)...")
raw_probe = torch.nn.Linear(ENC_NUM_PATCHES, 1).to(DEVICE)
raw_opt = torch.optim.AdamW(raw_probe.parameters(), lr=1e-3)
raw_train = torch.tensor([w[:, xlk_col] for w in train_w], dtype=torch.float32)
raw_val = torch.tensor([w[:, xlk_col] for w in val_w], dtype=torch.float32)
best_raw = 0.0
for epoch in range(NUM_PROBE_EPOCHS):
    raw_probe.train()
    perm = torch.randperm(len(raw_train))
    for i in range(0, len(perm), 32):
        idx = perm[i:i + 32]
        x = raw_train[idx].to(DEVICE)
        y = train_labels[idx].to(DEVICE).unsqueeze(1)
        raw_opt.zero_grad()
        loss = bce(raw_probe(x), y)
        loss.backward()
        raw_opt.step()
    raw_probe.eval()
    with torch.no_grad():
        acc = (raw_probe(raw_val.to(DEVICE)).squeeze() > 0).float().eq(val_labels.to(DEVICE)).float().mean().item()
    if acc > best_raw:
        best_raw = acc
print(f"   Raw returns probe best val acc: {best_raw:.3f}")


# -- 2d. Probe training curves plot -----------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, NUM_PROBE_EPOCHS + 1), probe_train_losses, label='train')
ax1.plot(range(1, NUM_PROBE_EPOCHS + 1), probe_val_losses, label='val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BCE Loss')
ax1.set_title('Probe Training Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, NUM_PROBE_EPOCHS + 1), probe_train_accs, label='train')
ax2.plot(range(1, NUM_PROBE_EPOCHS + 1), probe_val_accs, label='val')
ax2.axhline(y=baseline_acc, color='gray', linestyle='--', label='majority baseline')
if best_raw > 0:
    ax2.axhline(y=best_raw, color='green', linestyle='--', label='raw returns baseline')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Probe Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/probe_curves.png", dpi=150)
plt.close()
print(f"Probe curves saved to {OUTPUT_DIR}/probe_curves.png")


# ============================================================================
# SECTION 3: UNIFIED VERDICT
# ============================================================================
print(f"\n{'═' * 60}")
print(f"SECTION 3: Unified Evaluation Report")
print(f"{'═' * 60}")

lines = []
lines.append("JEPA Embedding Evaluation Report")
lines.append("=" * 40)
lines.append(f"Model: {MODEL_PATH}")
lines.append("")
lines.append("--- Embedding Structure ---")
lines.append(f"  Mean pairwise cos (train): {mean_train_sim:.4f}  +/- {std_train_sim:.4f}")
lines.append(f"  Mean pairwise cos (test):  {mean_test_sim:.4f}  +/- {std_test_sim:.4f}")
lines.append(f"  Effective dims (90% var):  {n_90} / {train_emb.shape[1]}")
lines.append(f"  Effective dims (95% var):  {n_95} / {train_emb.shape[1]}")
lines.append(f"  Mean shift train->test:    {mean_shift:.4f}")
lines.append(f"  Cosine verdict:            {collapse_msg}")
lines.append(f"  PCA verdict:               {pca_msg}")
lines.append("")
lines.append("--- Linear Probe ---")
lines.append(f"  Probe ticker:              {PROBE_TICKER}")
lines.append(f"  Majority baseline acc:     {baseline_acc:.3f}")
lines.append(f"  JEPA probe best val acc:   {best_val_acc:.3f}")
lines.append(f"  Raw returns probe acc:     {best_raw:.3f}")
lines.append(f"  Probe verdict:             {probe_msg}")
lines.append("")

if mean_train_sim < 0.85 and n_90 > 5 and best_val_acc > baseline_acc + 0.03:
    final = "PASS: Encoder learned real, diverse representations with market signal. Ready for GRPO."
elif mean_train_sim < 0.95 and n_90 > 3 and best_val_acc > baseline_acc:
    final = "MARGINAL: Representations may be useful but room for improvement."
else:
    final = "FAIL: Possible collapse or no detectable market signal. Check training."

lines.append(f"FINAL VERDICT: {final}")
lines.append("=" * 40)

for line in lines:
    print(line)

report_path = os.path.join(OUTPUT_DIR, "report.txt")
with open(report_path, 'w') as f:
    f.write('\n'.join(lines))
print(f"Report saved to {report_path}")

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
