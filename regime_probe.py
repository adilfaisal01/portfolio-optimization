"""Regime classifier probe: do JEPA embeddings encode market regime?"""
import sys, os


import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from src.models.encoder import Encoder
from individual_stocks.data_class_parquet import StockMarketJEPADataset

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cpu"
PARQUET_PATH = "/mnt/E/github-projects/portfolio-optimization/individual_stocks/parquet_data/sector_etf_clean_testingset.parquet"
NUM_PATCHES = 20
CONTEXT = 10

# ---- Load encoder ----
encoder = Encoder(dim_in=49, num_patches=NUM_PATCHES, kernel_size=49,
                  embed_dim=64, embed_bias=True, nhead=8, jepa=True, num_layers=4)
enc_state = torch.load(
    "/mnt/E/github-projects/portfolio-optimization/jepa-model/jepa_model_12/model_model_epoch_1000_encoder.pt",
    map_location="cpu", weights_only=True)
encoder.load_state_dict(enc_state)
encoder.to(DEVICE)
encoder.eval()
for p in encoder.parameters(): p.requires_grad = False

# ---- Build dataset and extract embeddings ----
base_ds = StockMarketJEPADataset(
    mask_ratio=0.0, num_patches=NUM_PATCHES, vix_fairweather=20,
    parquet_path=PARQUET_PATH,
)

df = pd.read_parquet(PARQUET_PATH)
dates = df['date'].dropna().unique()
dates = np.sort(dates)
NUM_WINDOWS = len(dates) // NUM_PATCHES

# Map windows to years
window_dates = []
for i in range(NUM_WINDOWS):
    start = i * NUM_PATCHES
    win_dates = dates[start:start + NUM_PATCHES]
    window_dates.append((win_dates[0], win_dates[-1]))

year_map = {}
for i, (start, end) in enumerate(window_dates):
    year = pd.Timestamp(start).year
    year_map[i] = year

# Extract embeddings for all windows
print("Extracting embeddings for all windows...")
embeddings = []
labels = []
window_ids = []

for win_idx in range(NUM_WINDOWS):
    window, _, _ = base_ds[win_idx]
    window = window.unsqueeze(0).to(DEVICE)  # [1, 20, 49]
    
    ctx = window[:, :CONTEXT, :].contiguous()
    with torch.no_grad():
        z = encoder(ctx)  # [1, 10, 64]
    
    # Average over time dimension to get one embedding per window
    z_avg = z.mean(dim=1).squeeze(0)  # [64]
    
    embeddings.append(z_avg.cpu().numpy())
    labels.append(year_map[win_idx])
    window_ids.append(win_idx)

embeddings = np.array(embeddings)  # [N, 64]
labels = np.array(labels)

print(f"Total windows: {len(embeddings)}")
print(f"Embedding dim: {embeddings.shape[1]}")
print(f"Years: {sorted(np.unique(labels))}")
print()

# ---- Train linear classifier (logistic regression) ----
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Standardize
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Cross-validated accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
cv_scores = cross_val_score(lr, embeddings_scaled, labels, cv=cv, scoring='accuracy')

print("=" * 60)
print("REGIME CLASSIFIER: Linear Probe on JEPA Embeddings")
print("=" * 60)
print(f"5-fold CV accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print(f"Per-fold: {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"Random chance (6 classes): {100/6:.1f}%")
print()

# Train on full data for confusion matrix
lr.fit(embeddings_scaled, labels)
preds = lr.predict(embeddings_scaled)
train_acc = accuracy_score(labels, preds)

print(f"Full-train accuracy: {train_acc*100:.1f}%")
print()
print("Confusion matrix (rows=actual, cols=predicted):")
years = sorted(np.unique(labels))
cm = confusion_matrix(labels, preds, labels=years)
print(f"{'':>8}", end="")
for y in years:
    print(f"{y:>8}", end="")
print()
for i, y in enumerate(years):
    print(f"{y:>8}", end="")
    for j in range(len(years)):
        print(f"{cm[i,j]:>8}", end="")
    print()

print()
print("Per-class metrics:")
print(classification_report(labels, preds, target_names=[str(y) for y in years]))

# ---- Compare to baseline: always predict majority class ----
from collections import Counter
majority_class = Counter(labels).most_common(1)[0][0]
baseline_acc = (labels == majority_class).mean()
print(f"Baseline (always predict {majority_class}): {baseline_acc*100:.1f}%")

# ---- Try with per-patch embeddings (not averaged) ----
print("\n" + "=" * 60)
print("PER-PATCH ANALYSIS: Does each patch embedding carry regime info?")
print("=" * 60)

# Extract per-patch embeddings
patch_embeddings = []  # [N*10, 64]
patch_labels = []
for win_idx in range(NUM_WINDOWS):
    window, _, _ = base_ds[win_idx]
    window = window.unsqueeze(0).to(DEVICE)
    ctx = window[:, :CONTEXT, :].contiguous()
    with torch.no_grad():
        z = encoder(ctx)  # [1, 10, 64]
    for t in range(CONTEXT):
        patch_embeddings.append(z[0, t].cpu().numpy())
        patch_labels.append(year_map[win_idx])

patch_embeddings = np.array(patch_embeddings)
patch_labels = np.array(patch_labels)

scaler2 = StandardScaler()
patch_scaled = scaler2.fit_transform(patch_embeddings)

cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr2 = LogisticRegression(max_iter=1000, random_state=42)
cv_scores2 = cross_val_score(lr2, patch_scaled, patch_labels, cv=cv2, scoring='accuracy')

print(f"Per-patch 5-fold CV accuracy: {cv_scores2.mean()*100:.1f}% ± {cv_scores2.std()*100:.1f}%")
print(f"Random chance (6 classes): {100/6:.1f}%")
