# FILE: jepa/embedding_db/cost_fcn.py
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
from jepa.models.encoder import Encoder
from jepa.data.data_class_parquet import StockMarketJEPADataset
from jepa.data.dataextraction import DataExtractor
from torch.utils.data import DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH     = "jepa-model/jepa_model_10/model_epoch_2000.pt"
TRAIN_PARQUET  = "jepa/data/parquet_data/sector_etf_clean_trainingset.parquet"
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
# 1. GET DATES (just the date index, no tensor rebuild)
# ============================================================================
print("=" * 60)
print("Building embedding database")
print("=" * 60)

de = DataExtractor(ticker_list=None, macro_indices=None, dataset=TRAIN_PARQUET)
etf_lr, _, _ = de.get_assets()
dates_all = etf_lr.index
print(f"Total days: {len(dates_all)}")
print(f"Date range: {dates_all[0]} to {dates_all[-1]}")


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
# 3. ITERATE DATASET → EMBEDDINGS + VIX + DATES
# ============================================================================
dataset = StockMarketJEPADataset(
    mask_ratio=0.0, num_patches=NUM_PATCHES,
    vix_fairweather=VIX_FAIRWEATHER, parquet_path=TRAIN_PARQUET,
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
num_windows = len(dataset)
print(f"Number of {NUM_PATCHES}-day windows: {num_windows}")

start_dates = []
end_dates = []
embeddings = []
vix_avgs = []

for i, batch in enumerate(loader):
    x = batch[0].to(DEVICE)  # (1, 20, 49)

    with torch.no_grad():
        z = encoder(x, mask=None)  # (1, 20, 64)

    emb = z.flatten(start_dim=1).cpu()  # (1, 1280)

    # VIX is column 48 in the window tensor
    vix_avg = x[0, :, 48].mean().item()

    start = i * NUM_PATCHES
    end = start + NUM_PATCHES

    start_dates.append(dates_all[start])
    end_dates.append(dates_all[end - 1])
    embeddings.append(emb)
    vix_avgs.append(vix_avg)

embeddings = torch.cat(embeddings, dim=0)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Sample window 0: {start_dates[0]} → {end_dates[0]}, VIX={vix_avgs[0]:.3f}")


# ============================================================================
# 4. SAVE
# ============================================================================
db = {
    "start_dates": [str(d) for d in start_dates],
    "end_dates": [str(d) for d in end_dates],
    "embeddings": embeddings,
    "vix_avg": torch.tensor(vix_avgs),
}
torch.save(db, os.path.join(OUTPUT_DIR, "embedding_db.pt"))
print(f"\nSaved: {OUTPUT_DIR}/embedding_db.pt")

df = pd.DataFrame({
    "start_date": [str(d) for d in start_dates],
    "end_date": [str(d) for d in end_dates],
    "vix_avg": vix_avgs,
})
df["embedding"] = [emb.numpy().tolist() for emb in embeddings]
df.to_parquet(os.path.join(OUTPUT_DIR, "embedding_db.parquet"))
print(f"Saved: {OUTPUT_DIR}/embedding_db.parquet")

print("\n✅ Embedding database built!")
print(f"   {num_windows} windows × 1280 dims")
print(f"   Date range: {start_dates[0]} → {end_dates[-1]}")
