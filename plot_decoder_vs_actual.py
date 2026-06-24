# FILE: plot_decoder_vs_actual.py (85 lines)
"""
Plot decoder predictions vs actual data on the test set.
Uses normalized decoder, shows both normalized and un-normalized views.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from src.models.encoder import Encoder
from src.models.decoder import LinearDecoder
from individual_stocks.data_class_parquet import StockMarketJEPADataset

# --- Config ----------------------------------------------------------------
ENC_PATH       = "jepa-model/jepa_model_10/model_epoch_2000.pt"
DECODER_PATH   = "jepa-model/decoder_epoch_2000.pt"
NORM_PATH      = "jepa-model/decoder_norm_stats.pt"
TEST_PARQUET   = "individual_stocks/parquet_data/sector_etf_clean_testingset.parquet"
DEVICE         = "cpu"
BATCH_SIZE     = 1

ENC_DIM_IN      = 49
ENC_NUM_PATCHES = 20
ENC_KERNEL_SIZE = 49
ENC_EMBED_DIM   = 64
ENC_NHEAD       = 8
ENC_NUM_LAYERS  = 4

# --- Load models -----------------------------------------------------------
encoder = Encoder(
    dim_in=ENC_DIM_IN, num_patches=ENC_NUM_PATCHES, kernel_size=ENC_KERNEL_SIZE,
    embed_dim=ENC_EMBED_DIM, embed_bias=True, nhead=ENC_NHEAD,
    jepa=True, num_layers=ENC_NUM_LAYERS,
)
encoder.load_state_dict(torch.load(ENC_PATH, map_location="cpu", weights_only=True))
encoder.eval()

decoder = LinearDecoder(emb_dim=ENC_EMBED_DIM, patch_size=ENC_DIM_IN)
decoder.load_state_dict(torch.load(DECODER_PATH, map_location="cpu", weights_only=True))
decoder.eval()

norm_stats = torch.load(NORM_PATH, map_location="cpu", weights_only=True)
feat_mean = norm_stats["mean"]
feat_std  = norm_stats["std"]

def normalize(w):
    return (w - feat_mean) / feat_std

def unnormalize(w):
    return w * feat_std + feat_mean

# --- Data ------------------------------------------------------------------
test_dataset = StockMarketJEPADataset(
    mask_ratio=0.7, num_patches=ENC_NUM_PATCHES, vix_fairweather=20,
    parquet_path=TEST_PARQUET,
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Inference -------------------------------------------------------------
all_actual = []
all_pred   = []

with torch.no_grad():
    for window, masks, non_masks in test_loader:
        window_norm = normalize(window)
        all_indices = torch.arange(ENC_NUM_PATCHES).unsqueeze(0)
        embeddings = encoder(window, all_indices)
        pred_norm = decoder(embeddings)
        all_actual.append(window.numpy())
        all_pred.append(unnormalize(pred_norm).numpy())

all_actual = np.concatenate(all_actual, axis=0)
all_pred   = np.concatenate(all_pred, axis=0)

# --- Plot: ETF returns only (indices 0-10) --------------------------------
# 11 sector ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XRT, VNQ
etf_names = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XRT", "VNQ"]
n_etfs = len(etf_names)
n_windows = 3

fig, axes = plt.subplots(n_etfs, n_windows, figsize=(4 * n_windows, 2 * n_etfs))

for col, win_idx in enumerate([0, 10, 20]):
    for row in range(n_etfs):
        ax = axes[row, col]
        actual = all_actual[win_idx, :, row]
        pred   = all_pred[win_idx, :, row]
        ax.plot(range(ENC_NUM_PATCHES), actual, 'b-o', label='actual', markersize=3, linewidth=1)
        ax.plot(range(ENC_NUM_PATCHES), pred,   'r--s', label='pred', markersize=3, linewidth=1)
        if col == 0:
            ax.set_ylabel(etf_names[row], fontsize=8)
        if row == 0:
            ax.set_title(f"Window {win_idx}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
        if row == n_etfs - 1:
            ax.set_xlabel("Patch (time step)", fontsize=8)
        if col == n_windows - 1 and row == 0:
            ax.legend(fontsize=7)

plt.suptitle("ETF Returns: Actual vs Decoder Reconstruction", fontsize=11)
plt.tight_layout()
plt.savefig("jepa-model/decoder_vs_actual_returns.png", dpi=150)
print("Saved to jepa-model/decoder_vs_actual_returns.png")

# --- Plot: VIX + macro returns (indices 33-37: GLD, USO, IYR, SHV, TIP) ---
macro_names = ["GLD", "USO", "IYR", "SHV", "TIP"]
fig2, axes2 = plt.subplots(len(macro_names) + 1, n_windows, figsize=(4 * n_windows, 2 * (len(macro_names) + 1)))

for col, win_idx in enumerate([0, 10, 20]):
    # VIX (index 48)
    ax = axes2[0, col]
    actual = all_actual[win_idx, :, 48]
    pred   = all_pred[win_idx, :, 48]
    ax.plot(range(ENC_NUM_PATCHES), actual, 'b-o', label='actual', markersize=3, linewidth=1)
    ax.plot(range(ENC_NUM_PATCHES), pred,   'r--s', label='pred', markersize=3, linewidth=1)
    if col == 0:
        ax.set_ylabel("VIX", fontsize=8)
    if col == n_windows - 1:
        ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # Macro returns (indices 33-37)
    for row, m_idx in enumerate(range(33, 38)):
        ax = axes2[row + 1, col]
        actual = all_actual[win_idx, :, m_idx]
        pred   = all_pred[win_idx, :, m_idx]
        ax.plot(range(ENC_NUM_PATCHES), actual, 'b-o', label='actual', markersize=3, linewidth=1)
        ax.plot(range(ENC_NUM_PATCHES), pred,   'r--s', label='pred', markersize=3, linewidth=1)
        if col == 0:
            ax.set_ylabel(macro_names[row], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
        if row == len(macro_names) - 1:
            ax.set_xlabel("Patch (time step)", fontsize=8)

plt.suptitle("VIX & Macro Returns: Actual vs Decoder Reconstruction", fontsize=11)
plt.tight_layout()
plt.savefig("jepa-model/decoder_vs_actual_macro.png", dpi=150)
print("Saved to jepa-model/decoder_vs_actual_macro.png")
