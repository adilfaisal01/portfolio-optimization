import torch
from src.models.encoder import Encoder
from torch.utils.data import DataLoader
import os
from individual_stocks.data_class_parquet import StockMarketJEPADataset
MODEL_PATH    = "jepa-model/jepa_model_10/model_epoch_2000.pt"
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
    jepa=False,
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

#training data loaded

train_dataset = StockMarketJEPADataset(
    mask_ratio=0.0, num_patches=20, vix_fairweather=20,
    parquet_path=TRAIN_PARQUET,
)

#running the encoder through the training data
data_loaded_train= DataLoader(dataset=train_dataset, batch_size=1,shuffle=False)

embeds_all_train=[]
for batch in data_loaded_train:
    x=batch[0].to(DEVICE)
    with torch.no_grad():
        z=encoder(x,mask=None)
    embeds_all_train.append(z)

embeds_all_train=torch.cat(embeds_all_train,dim=0)

print(embeds_all_train.shape) #(163,20,64) shape

