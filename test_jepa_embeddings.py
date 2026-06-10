import os
import sys
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from src.models.encoder import Encoder
from individual_stocks.data_class_parquet import StockMarketJEPADataset
from individual_stocks.dataextraction import DataExtractor

# --- Config ----------------------------------------------------------------
MODEL_PATH    = "jepa-model/model_epoch_50.pt"
TRAIN_PARQUET = "individual_stocks/parquet_data/sector_etf_clean_trainingset.parquet"
TEST_PARQUET  = "individual_stocks/parquet_data/sector_etf_clean_testingset.parquet"
OUTPUT_DIR    = "jepa-model/analysis/iteration-4"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_TICKER  = "VNQ"
NUM_PROBE_EPOCHS = 50

ENC_DIM_IN        = 49
ENC_NUM_PATCHES   = 20
ENC_KERNEL_SIZE   = 49
ENC_EMBED_DIM     = 256
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

train_dataset = StockMarketJEPADataset(
    mask_ratio=0.2, num_patches=20, vix_fairweather=20,
    parquet_path=TRAIN_PARQUET,
)
test_dataset = StockMarketJEPADataset(
    mask_ratio=0.2, num_patches=20, vix_fairweather=20,
    parquet_path=TEST_PARQUET,
)
print(f"Train windows (non-overlapping): {len(train_dataset)}")
print(f"Test windows (non-overlapping):  {len(test_dataset)}")

loader_train_data=DataLoader(dataset=train_dataset, batch_size=1,shuffle=False)
print(loader_train_data[0])

#vic-reg evaluation

# def vicreg_test(z:torch.Tensor, gamma=1.0, epsilon:float=1e-5):
#     N,D= z.shape
#     # variance--> V
#     std= torch.sqrt(torch.var(z))  
    
    