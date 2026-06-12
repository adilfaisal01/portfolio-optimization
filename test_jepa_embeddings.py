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
MODEL_PATH    = "jepa-model/jepa_model_12/workspace/outputs/jepa_model_11/model_model_epoch_2000_encoder.pt"
TRAIN_PARQUET = "individual_stocks/parquet_data/sector_etf_clean_trainingset.parquet"
TEST_PARQUET  = "individual_stocks/parquet_data/sector_etf_clean_testingset.parquet"
OUTPUT_DIR    = "jepa-model/analysis/iteration-1"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_TICKER  = "VNQ"
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

train_dataset = StockMarketJEPADataset(
    mask_ratio=0.7, num_patches=20, vix_fairweather=20,
    parquet_path=TRAIN_PARQUET,
)
test_dataset = StockMarketJEPADataset(
    mask_ratio=0.7, num_patches=20, vix_fairweather=20,
    parquet_path=TEST_PARQUET,
)
print(f"Train windows (non-overlapping): {len(train_dataset)}")
print(f"Test windows (non-overlapping):  {len(test_dataset)}")

loader_train_data=DataLoader(dataset=test_dataset, batch_size=1,shuffle=False)    
#vic-reg evaluation--> used Bardes, Ponce, LeCun ICLR 2022

def vicreg_eval(z:torch.Tensor, gamma=1.0, epsilon:float=1e-5):
     N,D= z.shape #N: batch_size, D: dimension of the representations
     # variance --> V
     std= torch.sqrt(z.var(dim=0)+epsilon)
     variance_loss= torch.mean(torch.relu(gamma-std))
     #covariance --> C
     center_z= z-z.mean(dim=0)
     covar= (center_z.T@center_z)/(N-1)
     diagonal_covar= covar*torch.eye(D)
     off_diag_covar= covar-diagonal_covar
     covar_loss=(off_diag_covar**2).sum()/D
     # SVD breakdown
     u,s,vh= torch.linalg.svd(center_z, full_matrices= False)
     p= s/s.sum()
     effective_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-10)))

     ## max auto corr (off diagonal)    
     corr=covar/(torch.outer(std,std)+epsilon)
     max_corr = corr[~torch.eye(D, dtype=torch.bool)].abs().max()

     return {
             'variance_loss': variance_loss.item(),
             'covariance_loss': covar_loss.item(),
             'mean_std': std.mean().item(),
             'min_std': std.min().item(),
             'effective_rank': effective_rank.item(),
             'max_corr': max_corr.item(),

      }

## inference--> collect all the embeddings then concat then for vicreg evaluation
embedds_all=[]
for batch in loader_train_data:
    x=batch[0].to(DEVICE)
    non_masks=batch[2].to(DEVICE)
    with torch.no_grad():
        z=encoder(x,non_masks)
    embedds_all.append(z)
embedds_all=torch.cat(embedds_all, dim=0)
print(embedds_all.shape)

results=[]
#day by day breakdown for vicreg loss
for t in range(embedds_all.shape[1]):
    z_t=embedds_all[:,t,:] ##[batch_size, dimensions]
    m=vicreg_eval(z_t)
    results.append(m)
    print(f"{t:4d}  {m['variance_loss']:>9}  {m['covariance_loss']:>9}  "
          f"{m['mean_std']:>8}  {m['min_std']:>7}  {m['effective_rank']:>8}  {m['max_corr']:>8}")


for k in results[0]:
    vals = [r[k] for r in results]
    print(f"  {k:20s}:  mean={np.mean(vals):>8.4f}  min={np.min(vals):>8.4f}  max={np.max(vals):>8.4f}")

print()

# Pass / Fail
avg_var_loss = np.mean([r['variance_loss'] for r in results])
avg_cov_loss = np.mean([r['covariance_loss'] for r in results])
avg_eff_rank = np.mean([r['effective_rank'] for r in results])