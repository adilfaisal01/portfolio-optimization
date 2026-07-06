 #test_set query

import torch
from jepa.models.encoder import Encoder
from jepa.data.data_class_parquet import StockMarketJEPADataset
from torch.utils.data import DataLoader
from jepa.embedding_db.query import query_similar

# # Load encoder
encoder = Encoder(dim_in=49, num_patches=20, kernel_size=49, embed_dim=64,
                   embed_bias=True, nhead=8, jepa=False, num_layers=4)
state = torch.load("jepa-model/jepa_model_10/model_epoch_2000.pt", map_location="cpu", weights_only=True)
encoder.load_state_dict(state)
encoder.eval()

# # Load test set
test_dataset = StockMarketJEPADataset(
     mask_ratio=0.0, num_patches=20, vix_fairweather=20,
     parquet_path="jepa/data/parquet_data/sector_etf_clean_testingset.parquet",
 )
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # March 2020 COVID crash window (index 2)
for i, batch in enumerate(test_loader):
     if i == 2:
         x = batch[0]
         break

with torch.no_grad():
     z = encoder(x, mask=None)

query_emb = z.flatten(start_dim=1).squeeze(0)  # (1280,)

start_date, end_date,