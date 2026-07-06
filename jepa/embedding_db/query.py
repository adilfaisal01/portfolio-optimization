import torch
import pandas as pd

# loading the parquet file for human readability
xy= pd.read_parquet("jepa-model/analysis/iteration-1/embedding_db.parquet")

# loading the machine readable dataset
db= torch.load("
    jepa-model/analysis/iteration-1/embedding_db.pt", weights_only=True)
print(db)