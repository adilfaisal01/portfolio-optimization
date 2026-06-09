#!/usr/bin/env python3
"""
test_jepa_embeddings.py — Sanity check for JEPA encoder
Tests if the trained encoder actually learned meaningful representations
instead of collapsing to junk.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path

# --- Config ----------------------------------------------------------------
MODEL_PATH    = "jepa-model/model_epoch_20.pt"  # best val @ epoch 35, but 20 also works
TRAIN_PARQUET = "individual_stocks/parquet_data/sector_etf_clean_trainingset.parquet"
TEST_PARQUET  = "individual_stocks/parquet_data/sector_etf_clean_testingset.parquet"
OUTPUT_DIR    = "jepa-model/analysis"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# JEPA architecture config  (must match what was used during training)
ENC_DIM_IN        = 49
ENC_NUM_PATCHES   = 20
ENC_KERNEL_SIZE   = 49
ENC_EMBED_DIM     = 256
ENC_NHEAD         = 8
ENC_NUM_LAYERS    = 4

sys.path.insert(0, '.')
from src.models.encoder import Encoder
from individual_stocks.data_class_parquet import StockMarketJEPADataset


# --- 1. Load model ---------------------------------------------------------
print("=" * 60)
print("🔍 JEPA Embedding Sanity Check")
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
print(f"✅ Loaded encoder from {MODEL_PATH}")
print(f"   Total params: {sum(p.numel() for p in encoder.parameters()):,}")


# --- 2. Load datasets & extract embeddings ---------------------------------
dataset = StockMarketJEPADataset(
    mask_ratio=0.2, num_patches=20, vix_fairweather=20,
    parquet_path=TRAIN_PARQUET,
)

test_dataset = StockMarketJEPADataset(
    mask_ratio=0.2, num_patches=20, vix_fairweather=20,
    parquet_path=TEST_PARQUET,
)

print(f"📦 Train windows: {len(dataset)}")
print(f"📦 Test windows:  {len(test_dataset)}")

@torch.no_grad()
def get_embeddings(ds, max_windows=None):
    """Run full windows through encoder, return [N, 20, 256] embeddings."""
    windows = []
    n = len(ds) if max_windows is None else min(len(ds), max_windows)
    for i in range(n):
        window, _, _ = ds[i]
        windows.append(window.unsqueeze(0))  # [1, 20, 49]
    x = torch.cat(windows, dim=0).to(DEVICE)
    emb = encoder(x)  # [N, 20, 256]
    # Mean pool over patches → one vector per window [N, 256]
    pooled = emb.mean(dim=1)
    return pooled.cpu().numpy()

train_emb = get_embeddings(dataset, max_windows=500)
test_emb  = get_embeddings(test_dataset, max_windows=500)
print(f"✅ Extracted {train_emb.shape[0]} train embeddings, {test_emb.shape[0]} test embeddings")
print(f"   Embedding dim: {train_emb.shape[1]}")


# --- 3. Collapse check: pairwise cosine similarity -------------------------
def cosine_similarity_matrix(emb):
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = emb_norm @ emb_norm.T
    return sim

train_sim = cosine_similarity_matrix(train_emb)
test_sim  = cosine_similarity_matrix(test_emb)

# Upper triangle (excluding diagonal)
triu_idx = np.triu_indices_from(train_sim, k=1)
triu_idx_test=np.triu_indices_from(test_sim, k=1)
mean_train_sim = train_sim[triu_idx].mean()
std_train_sim  = train_sim[triu_idx].std()
mean_test_sim  = test_sim[triu_idx_test].mean()
std_test_sim   = test_sim[triu_idx].std()

print(f"\n{'─' * 60}")
print(f"📐 Embedding Similarity Analysis")
print(f"{'─' * 60}")
print(f"   Train set — mean pairwise cos: {mean_train_sim:.4f}  ± {std_train_sim:.4f}")
print(f"   Test set  — mean pairwise cos: {mean_test_sim:.4f}  ± {std_test_sim:.4f}")

if mean_train_sim > 0.95:
    print("   ⚠️  WARNING: >0.95 similarity — likely representation collapse!")
    print("   The encoder maps everything to nearly the same embedding.")
elif mean_train_sim > 0.85:
    print("   🟡 CAUTION: >0.85 similarity — embeddings are quite similar.")
    print("   Might still be useful, but check t-SNE for structure.")
else:
    print("   ✅ Embeddings are diverse — encoder is distinguishing states.")


# --- 4. Effective rank analysis (PCA rank) ---------------------------------
pca = PCA(n_components=min(50, train_emb.shape[1]))
pca.fit(train_emb)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_90 = np.searchsorted(cum_var, 0.90) + 1
n_95 = np.searchsorted(cum_var, 0.95) + 1

print(f"\n📊 PCA on train embeddings:")
print(f"   Dims to explain 90% variance: {n_90}  / {train_emb.shape[1]}")
print(f"   Dims to explain 95% variance: {n_95}  / {train_emb.shape[1]}")

if n_90 <= 3:
    print("   ⚠️  90% variance in ≤3 dims — likely rank collapse.")
else:
    print("   ✅ Embeddings use >3 effective dimensions — real structure.")


# --- 5. Train vs Test distribution shift (MMD approximation) ---------------
# Compare embedding means & variances between train and test
train_mean = train_emb.mean(axis=0)
test_mean  = test_emb.mean(axis=0)
mean_shift = np.linalg.norm(train_mean - test_mean)
print(f"\n📏 Train/Test distribution shift:")
print(f"   Mean embedding shift (L2): {mean_shift:.4f}")

# Compare variance explained
pca_test = PCA(n_components=min(50, test_emb.shape[1]))
pca_test.fit(test_emb)
test_90 = np.searchsorted(np.cumsum(pca_test.explained_variance_ratio_), 0.90) + 1
print(f"   Test set: dims to explain 90% variance: {test_90} / {test_emb.shape[1]}")
print(f"   Train set: dims to explain 90% variance: {n_90} / {train_emb.shape[1]}")


# --- 6. t-SNE visualization ------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subset for t-SNE speed
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
print(f"\n🖼️  t-SNE plot saved to {OUTPUT_DIR}/tsne_embeddings.png")


# --- 7. Summary verdict ----------------------------------------------------
print(f"\n{'═' * 60}")
print(f"📋 SUMMARY")
print(f"{'═' * 60}")
print(f"   Mean pairwise cos (train): {mean_train_sim:.4f}")
print(f"   Mean pairwise cos (test):  {mean_test_sim:.4f}")
print(f"   Effective dims (90% var): {n_90}")
print(f"   Mean shift train→test:    {mean_shift:.4f}")

if mean_train_sim < 0.85 and n_90 > 5:
    print(f"\n   ✅ VERDICT: Encoder learned real representations.")
    print(f"      Embeddings are diverse, not collapsed. Ready for GRPO.")
elif mean_train_sim < 0.95 and n_90 > 3:
    print(f"\n   🟡 VERDICT: Likely useful, but room for improvement.")
    print(f"      Try reducing embed_dim or increasing num_patches.")
else:
    print(f"\n   ⚠️  VERDICT: Possible collapse or low-rank representation.")
    print(f"      Consider adding a regularization term or checking data.")