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
    idx=idx.tolist()
    return (
        [db["start_dates"][i] for i in idx],
        [db["end_dates"][i] for i in idx],
        db["vix_avg"][idx],
        [db['mean_returns'][i] for i in idx],
        [db['covariances'][i] for i in idx],
        scores
    )

# test_query->using the first historical as a test

test_query=db["embeddings"][0]
start_date, end_date, _,_,_,scores=query_similar(test_query)
print(f'Start-date: {start_date} \n\n',f'End-date:{end_date}\n\n', f'Score:{scores}')


# Sanity check: how spread out are the embeddings?
emb = db["embeddings"]
sims = torch.nn.functional.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=2)
mask = ~torch.eye(163, dtype=bool)
print(f"Min non-self similarity: {sims[mask].min().item():.4f}")
print(f"Max non-self similarity: {sims[mask].max().item():.4f}")
print(f"Mean non-self similarity: {sims[mask].mean().item():.4f}")