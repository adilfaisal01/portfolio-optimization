# FILE: train_return_probe.py (80 lines)
"""
Train a linear probe to extract ETF returns from JEPA embeddings.

The probe is a single Linear(64, 11) layer trained to predict
the 11 ETF log returns from the encoder embeddings.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from src.models.encoder import Encoder
from individual_stocks.data_class_parquet import StockMarketJEPADataset

# --- Config ----------------------------------------------------------------
ENC_PATH       = "jepa-model/jepa_model_10/model_epoch_2000.pt"
TRAIN_PARQUET  = "individual_stocks/parquet_data/sector_etf_clean_trainingset.parquet"
VAL_PARQUET    = "individual_stocks/parquet_data/sector_etf_clean_testingset.parquet"
PROBE_SAVE     = "jepa-model/return_probe_epoch_2000.pt"
DEVICE         = "cpu"
BATCH_SIZE     = 32
LR             = 1e-3
NUM_EPOCHS     = 200

ENC_DIM_IN      = 49
ENC_NUM_PATCHES = 20
ENC_KERNEL_SIZE = 49
ENC_EMBED_DIM   = 64
ENC_NHEAD       = 8
ENC_NUM_LAYERS  = 4
N_ASSETS        = 11  # ETF returns only (indices 0-10)

# --- Load encoder ----------------------------------------------------------
print(f"Loading encoder from {ENC_PATH} ...")
encoder = Encoder(
    dim_in=ENC_DIM_IN, num_patches=ENC_NUM_PATCHES, kernel_size=ENC_KERNEL_SIZE,
    embed_dim=ENC_EMBED_DIM, embed_bias=True, nhead=ENC_NHEAD,
    jepa=True, num_layers=ENC_NUM_LAYERS,
)
state = torch.load(ENC_PATH, map_location="cpu", weights_only=True)
encoder.load_state_dict(state)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False
print(f"  Total params: {sum(p.numel() for p in encoder.parameters()):,}")

# --- Data ------------------------------------------------------------------
train_dataset = StockMarketJEPADataset(
    mask_ratio=0.7, num_patches=ENC_NUM_PATCHES, vix_fairweather=20,
    parquet_path=TRAIN_PARQUET,
)
val_dataset = StockMarketJEPADataset(
    mask_ratio=0.7, num_patches=ENC_NUM_PATCHES, vix_fairweather=20,
    parquet_path=VAL_PARQUET,
)

# Compute return normalization stats
print("Computing return normalization stats...")
all_returns = []
for i in range(len(train_dataset)):
    window, _, _ = train_dataset[i]
    all_returns.append(window[:, :N_ASSETS])  # ETF returns only
all_returns = torch.cat(all_returns, dim=0)
ret_mean = all_returns.mean(dim=0, keepdim=True)
ret_std  = all_returns.std(dim=0, keepdim=True)
ret_std[ret_std < 1e-8] = 1.0
print(f"  Return means: {ret_mean.squeeze().numpy()}")
print(f"  Return stds:  {ret_std.squeeze().numpy()}")

def normalize_returns(r):
    return (r - ret_mean) / ret_std

# --- Probe ---------------------------------------------------------------
class MLPProbe(nn.Module):
    def __init__(self, emb_dim, n_assets, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, n_assets)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

probe = MLPProbe(ENC_EMBED_DIM, N_ASSETS, hidden_dim=32)
print(f"  Probe: MLP({ENC_EMBED_DIM} → 32 → {N_ASSETS})")

# --- Training --------------------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.AdamW(probe.parameters(), lr=LR)
mse_loss  = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    probe.train()
    train_loss = 0.0
    for window, masks, non_masks in train_loader:
        # Target: normalized ETF returns
        returns = window[:, :, :N_ASSETS]  # (B, 20, 11)
        returns_norm = normalize_returns(returns)

        # Encode ALL patches
        all_indices = torch.arange(ENC_NUM_PATCHES).unsqueeze(0).repeat(window.size(0), 1)
        with torch.no_grad():
            embeddings = encoder(window, all_indices)  # (B, 20, 64)

        # Predict returns from embeddings
        pred = probe(embeddings)  # (B, 20, 11)

        loss = mse_loss(pred, returns_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    probe.eval()
    val_loss = 0.0
    with torch.no_grad():
        for window, masks, non_masks in val_loader:
            returns = window[:, :, :N_ASSETS]
            returns_norm = normalize_returns(returns)
            all_indices = torch.arange(ENC_NUM_PATCHES).unsqueeze(0).repeat(window.size(0), 1)
            embeddings = encoder(window, all_indices)
            pred = probe(embeddings)
            val_loss += mse_loss(pred, returns_norm).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS}  |  train MSE: {train_loss:.6f}  |  val MSE: {val_loss:.6f}")

# --- Save ------------------------------------------------------------------
torch.save({
    "probe": probe.state_dict(),
    "ret_mean": ret_mean,
    "ret_std": ret_std,
}, PROBE_SAVE)
print(f"\nProbe saved to {PROBE_SAVE}")
