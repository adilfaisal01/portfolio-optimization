from src.models.encoder import Encoder
from src.models.utils.mask_utils import apply_mask
from src.models.predictor import Predictor
import copy
import torch
from individual_stocks.data_class_parquet import StockMarketJEPADataset
from torch.utils.data import DataLoader

dev= torch.device("cuda" if torch.cuda.is_available() else "cpu")

## loading the dataset
dataset=StockMarketJEPADataset(mask_ratio=0.6,num_patches=20,vix_fairweather=20,parquet_path="individual_stocks/parquet_data/sector_etf_clean_trainingset.parquet")
data_loaded= DataLoader(dataset,batch_size=1,shuffle=True)

# print(len(data_loaded.dataset[0][0]))

# ## setting up the definitions of the NNs
encoder= Encoder(dim_in=49,num_patches=20,kernel_size=49,embed_dim=256,embed_bias=True,nhead=8,jepa=True,num_layers=4)
ema_encoder= copy.deepcopy(encoder)
predictor= Predictor(num_patches=20,num_layers=2,nhead=4,predictor_embed_dim=512,encoder_embed_dim=256)

## move all networks to the correct device
encoder.to(dev)
predictor.to(dev)
ema_encoder.to(dev)
# ema encoder is not differentiated
for p in ema_encoder.parameters():
    p.requires_grad=False

# define which parameters need to be optimized--> encoder and predictor
params_encoder=encoder.parameters()
params_predictor=predictor.parameters()
all_params_opt=[params_encoder, params_predictor]
optimizer= torch.optim.AdamW(all_params_opt, lr=3e-4,weight_decay=1e-6)

#EMA scheduling definition
ema_scheduler= 





