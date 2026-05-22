import pandas as pd
from sklearn.decomposition import PCA

# reading the dataset
dataset=pd.read_parquet('final_dataset_stocks.parquet')
returns_pivot=dataset.pivot_table(values='log_return',index='date',columns='ticker').dropna(axis=1,thresh=2000).fillna(0)
spread_pivot=dataset.pivot_table(values='hl_spread',index='date',columns='ticker').dropna(axis=1,thresh=2000).fillna(0)
volume_pivot=dataset.pivot_table(values='RVOL',index='date',columns='ticker').dropna(axis=1,thresh=2000).fillna(0)

## doing PCA calculations

pca_returns=PCA(n_components=12,svd_solver='full')
print(pca_returns.fit(returns_pivot).explained_variance_ratio_)