from os import name

import torch
import pandas as pd

# # loading the parquet file for human readability
xy= pd.read_parquet("jepa-model/analysis/iteration-1/embedding_db.parquet")

# # loading the machine readable dataset
db= torch.load("jepa-model/analysis/iteration-1/embedding_db.pt", weights_only=True)

# # using top-K and similarity scores(via Euclidean distance) to find the closest embeddings to the appropriate historical window
def query_similar(current_embedding:torch.Tensor, k:int=10):
     embedding_store= db['embeddings']
     cosine_sim= torch.cdist(current_embedding.unsqueeze(0),embedding_store).squeeze(0)
     scores, idx= torch.topk(cosine_sim,k=k,largest=False)
     idx=idx.tolist()
     return (
         [db["start_dates"][i] for i in idx],
         [db["end_dates"][i] for i in idx],
         db["vix_avg"][idx],
         db['mean_returns'][idx],
         db['covariances'][idx],
         scores
     )

# test_query->using the first historical as a test, distance should be ~0 to self and massive to other periods
if name=='__main__':
    test_query=db["embeddings"][162]
    start_date, end_date, _,mean_returns,_,scores=query_similar(test_query)
    print(f'Start-date: {start_date} \n\n',f'End-date:{end_date}\n\n', f'Score:{mean_returns.shape}')