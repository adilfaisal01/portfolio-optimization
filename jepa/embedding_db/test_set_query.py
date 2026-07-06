 #test_set query

import torch
from jepa.models.encoder import Encoder
from jepa.data.data_class_parquet import StockMarketJEPADataset
from torch.utils.data import DataLoader
from jepa.embedding_db.query import query_similar,weighted_financial_metrics

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

print("COVID crash (Mar 9 → Apr 3, 2020) → top 5 historical matches:")
start_date, end_date,vix_avg,mean_returns, covars,scores=query_similar(query_emb, k=5) # find topk=5, 5 closest neighbors
print(covars.shape)  #(5,11,11)
print(mean_returns.shape) #(5,11)
# print(covars[0].shape)
# print(f'Start date:{start_date}\n\n', f'End_date:{end_date}\n\n', f'Scores: {scores} \n\n', f'Vix Averages: {torch.Tensor(mean_returns).shape}')

# implementing sparse attention (top K neighbors, via the normalized scores)
weighted_cv, weighted_mu=weighted_financial_metrics(query_emb,k=5,tau=0.4)
print(weighted_cv.shape)
print(weighted_mu.shape)