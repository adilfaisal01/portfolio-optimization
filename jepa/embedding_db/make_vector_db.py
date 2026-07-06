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
mean_log_returns=[]
covars=[]

for i, batch in enumerate(loader):
    x = batch[0].to(DEVICE)  # (1, 20, 49)

    with torch.no_grad():
        z = encoder(x, mask=None)  # (1, 20, 64)

    emb = z.flatten(start_dim=1)  # (1, 1280)

    # VIX is column 48 in the window tensor, Pulling log returns as well as h-l spreads
    vix_avg = x[0, :, 48].mean().item()
    log_ret_etfs= x[0,:,0:11]
    mu= log_ret_etfs.mean(dim=0)

    # finding average covariance using the Parkinson Estimator
    hl_spread=x[0,:,11:22]
    vol = hl_spread.mean(dim=0) / (2 * (2 / torch.pi) ** 0.5)  # (11,)
    corr = torch.corrcoef(log_ret_etfs.T)                        # (11, 11)
    cov = torch.diag(vol) @ corr @ torch.diag(vol)              # (11, 11)
    
    start = i * NUM_PATCHES
    end = start + NUM_PATCHES

    start_dates.append(dates_all[start])
    end_dates.append(dates_all[end - 1])
    embeddings.append(emb)
    vix_avgs.append(vix_avg)
    mean_log_returns.append(mu)
    covars.append(cov)

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
    "mean_returns":torch.stack(mean_log_returns),
    "covariances":torch.stack(covars),
}
torch.save(db, os.path.join(OUTPUT_DIR, "embedding_db.pt"))
print(f"\nSaved: {OUTPUT_DIR}/embedding_db.pt")

df = pd.DataFrame({
    "start_date": [str(d) for d in start_dates],
    "end_date": [str(d) for d in end_dates],
    "vix_avg": vix_avgs,
})
df["embedding"] = [emb.numpy().tolist() for emb in embeddings]
df["mean_log_returns"] = [m.tolist() for m in mean_log_returns]  # list of 11 floats
df["covar"] = [c.flatten().tolist() for c in covars]          # list of 121 floats
df.to_parquet(os.path.join(OUTPUT_DIR, "embedding_db.parquet"))
print(f"Saved: {OUTPUT_DIR}/embedding_db.parquet")

print("\n✅ Embedding database built!")
print(f"   {num_windows} windows × 1280 dims")
print(f"   Date range: {start_dates[0]} → {end_dates[-1]}")
