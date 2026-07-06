# FILE: jepa/embedding_db/cost_fcn.py (120 lines)
"""
Build and query the JEPA embedding database.

Produces:
  - jepa-model/analysis/iteration-1/embedding_db.pt  (torch save)
  - jepa-model/analysis/iteration-1/embedding_db.parquet  (human-readable)

Each entry: (start_date, end_date, embedding_1280, vix_avg)
"""
import torch
import os
import pandas as pd
import numpy as np
from jepa.data.data_class_parquet import StockMarketJEPADataset
from jepa.models.encoder import Encoder
from jepa.data.dataextraction import DataExtractor

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH     = "jepa-model/jepa_model_10/model_epoch_2000.pt"
TRAIN_PARQUET  = "jepa/data/parquet_data/sector_etf_clean_trainingset.parquet"
TEST_PARQUET   = "jepa/data/parquet_data/sector_etf_clean_testingset.parquet"
OUTPUT_DIR     = "jepa-model/analysis/iteration-1"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
VIX_FAIRWEATHER = 20
NUM_PATCHES    = 20

# ── Encoder config ─────────────────────────────────────────────────────────
ENC_DIM_IN      = 49
ENC_KERNEL_SIZE = 49
ENC_EMBED_DIM   = 64
ENC_NHEAD       = 8
ENC_NUM_LAYERS  = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 1. BUILD RAW DATA TENSOR + DATES (mirrors StockMarketJEPADataset internals)
# ============================================================================
print("=" * 60)
print("Building embedding database")
print("=" * 60)

dat,_,_= StockMarketJEPADataset
# Drop NaN rows (VIX column is index 48)
nan_mask = ~torch.isnan(dat).any(dim=1)
dat = dat[nan_mask]
dates_clean = dates_all[nan_mask.numpy()]

# VIX is the last column (index 48)
vix_col = dat[:, 48]

print(f"Total clean days: {len(dates_clean)}")
print(f"Date range: {dates_clean[0]} to {dates_clean[-1]}")


# ============================================================================
# 2. LOAD ENCODER
# ============================================================================
encoder = Encoder(
    dim_in=ENC_DIM_IN, num_patches=NUM_PATCHES,
    kernel_size=ENC_KERNEL_SIZE, embed_dim=ENC_EMBED_DIM,
    embed_bias=True, nhead=ENC_NHEAD, jepa=False,
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
# 3. SLIDE WINDOWS → EMBEDDINGS + VIX + DATES
# ============================================================================
num_windows = len(dates_clean) // NUM_PATCHES
print(f"Number of {NUM_PATCHES}-day windows: {num_windows}")

start_dates = []
end_dates = []
embeddings = []
vix_avgs = []

for i in range(num_windows):
    start = i * NUM_PATCHES
    end = start + NUM_PATCHES

    # Window of raw data
    window = dat[start:end].unsqueeze(0).to(DEVICE)  # (1, 20, 49)

    # Encode
    with torch.no_grad():
        z = encoder(window, mask=None)  # (1, 20, 64)

    # Flatten to 1280
    emb = z.flatten(start_dim=1).cpu()  # (1, 1280)

    # VIX average over the window (normalized, already /20)
    vix_avg = vix_col[start:end].mean().item()

    start_dates.append(dates_clean[start])
    end_dates.append(dates_clean[end - 1])
    embeddings.append(emb)
    vix_avgs.append(vix_avg)

# Stack embeddings → (num_windows, 1280)
embeddings = torch.cat(embeddings, dim=0)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Sample window 0: {start_dates[0]} → {end_dates[0]}, VIX={vix_avgs[0]:.3f}")


# ============================================================================
# 4. SAVE
# ============================================================================

# ── Torch format (for fast loading in cost function) ──
db = {
    "start_dates": [str(d) for d in start_dates],
    "end_dates": [str(d) for d in end_dates],
    "embeddings": embeddings,       # (N, 1280)
    "vix_avg": torch.tensor(vix_avgs),  # (N,)
}
torch.save(db, os.path.join(OUTPUT_DIR, "embedding_db.pt"))
print(f"\nSaved: {OUTPUT_DIR}/embedding_db.pt")

# ── Parquet format (for inspection) ──
df = pd.DataFrame({
    "start_date": [str(d) for d in start_dates],
    "end_date": [str(d) for d in end_dates],
    "vix_avg": vix_avgs,
})
# Embeddings as list of lists for parquet
df["embedding"] = [emb.numpy().tolist() for emb in embeddings]
df.to_parquet(os.path.join(OUTPUT_DIR, "embedding_db.parquet"))
print(f"Saved: {OUTPUT_DIR}/embedding_db.parquet")

print("\n✅ Embedding database built!")
print(f"   {num_windows} windows × 1280 dims")
print(f"   Date range: {start_dates[0]} → {end_dates[-1]}")

