# FILE: train_decoder.py (65 lines)
"""
Train the LinearDecoder to reconstruct raw features from JEPA embeddings.

Pipeline:
  1. Load trained encoder (M10@1000, embed_dim=64)
  2. Freeze encoder
  3. Encode all patches (no masking) → 64-dim embeddings
  4. Decode → 49-dim features
  5. MSE loss against original window
  6. Save decoder weights
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from src.models.encoder import Encoder
from src.models.decoder import LinearDecoder
from individual_stocks.data_class_parquet import StockMarketJEPADataset

# --- Config ----------------------------------------------------------------
ENC_PATH       = "jepa-model/jepa_model_10/model_epoch_2000.pt"
TRAIN_PARQUET  = "individual_stocks/parquet_data/sector_etf_clean_trainingset.parquet"
VAL_PARQUET    = "individual_stocks/parquet_data/sector_etf_clean_testingset.parquet"
DECODER_SAVE   = "jepa-model/decoder_epoch_2000.pt"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 32
LR             = 1e-3
NUM_EPOCHS     = 10

ENC_DIM_IN      = 49
ENC_NUM_PATCHES = 20
ENC_KERNEL_SIZE = 49
ENC_EMBED_DIM   = 64
ENC_NHEAD       = 8
ENC_NUM_LAYERS  = 4

# --- Load encoder ----------------------------------------------------------
print(f"Loading encoder from {ENC_PATH} ...")
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
state = torch.load(ENC_PATH, map_location="cpu", weights_only=True)
encoder.load_state_dict(state)
encoder.to(DEVICE)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False
print(f"  Total params: {sum(p.numel() for p in encoder.parameters()):,}")
print(f"  Device: {DEVICE}")

# --- Decoder ---------------------------------------------------------------
decoder = LinearDecoder(emb_dim=ENC_EMBED_DIM, patch_size=ENC_DIM_IN)
decoder.to(DEVICE)
print(f"  Decoder: Linear({ENC_EMBED_DIM} → {ENC_DIM_IN})")

# --- Data ------------------------------------------------------------------
train_dataset = StockMarketJEPADataset(
    mask_ratio=0.7, num_patches=ENC_NUM_PATCHES, vix_fairweather=20,
    parquet_path=TRAIN_PARQUET,
)
val_dataset = StockMarketJEPADataset(
    mask_ratio=0.7, num_patches=ENC_NUM_PATCHES, vix_fairweather=20,
    parquet_path=VAL_PARQUET,
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"  Train windows: {len(train_dataset)}")
print(f"  Val windows:   {len(val_dataset)}")

# --- Training --------------------------------------------------------------
optimizer = torch.optim.AdamW(decoder.parameters(), lr=LR)
mse_loss  = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    decoder.train()
    train_loss = 0.0
    for window, masks, non_masks in train_loader:
        window = window.to(DEVICE)
        # Encode ALL patches — pass all indices as non_masks
        all_indices = torch.arange(ENC_NUM_PATCHES).unsqueeze(0).repeat(window.size(0), 1).to(DEVICE)
        with torch.no_grad():
            embeddings = encoder(window, all_indices)  # (B, 20, 64)
        # Decode
        reconstructed = decoder(embeddings)  # (B, 20, 49)
        # Loss: MSE against original window
        loss = mse_loss(reconstructed, window)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    decoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for window, masks, non_masks in val_loader:
            window = window.to(DEVICE)
            all_indices = torch.arange(ENC_NUM_PATCHES).unsqueeze(0).repeat(window.size(0), 1).to(DEVICE)
            embeddings = encoder(window, all_indices)
            reconstructed = decoder(embeddings)
            val_loss += mse_loss(reconstructed, window).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS}  |  train MSE: {train_loss:.6f}  |  val MSE: {val_loss:.6f}")

# --- Save ------------------------------------------------------------------
torch.save(decoder.state_dict(), DECODER_SAVE)
print(f"\nDecoder saved to {DECODER_SAVE}")
