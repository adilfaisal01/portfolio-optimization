import torch
import pandas as pd

# loading the parquet file for human readability
xy= pd.read_parquet("jepa-model/analysis/iteration-1/embedding_db.parquet")

# loading the machine readable dataset
db= torch.load("jepa-model/analysis/iteration-1/embedding_db.pt", weights_only=True)

# using top-K and similarity scores to find the closest embeddings to the appropriate historical window
def query_similar(current_embedding:torch.Tensor, k:int=10):
    embedding_store= db['embeddings']
    cosine_sim= torch.nn.functional.cosine_similarity(current_embedding.unsqueeze(0),embedding_store)
    scores, idx= torch.topk(cosine_sim,k=k)

    return (
        [db["start_dates"][i] for i in idx],
        [db["end_dates"][i] for i in idx],
        db["vix_avg"][idx],
        db['mean_returns'][idx],
        db['covars'][idx],
        scores
    )

# test_query->using the first historical as a test

