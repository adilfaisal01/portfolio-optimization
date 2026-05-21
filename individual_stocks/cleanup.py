import pandas as pd
import numpy as np


class StockCleaner:
    def __init__(self, parquet_path: str):
        self.raw = pd.read_parquet(parquet_path)
        self.long = None

    def extract(self, price_types: list[str]) -> 'StockCleaner':
        needed = list(dict.fromkeys(['Close'] + price_types))
        subset = self.raw.loc[:, pd.IndexSlice[needed, :]]
        long = subset.stack(level='Ticker').reset_index()
        long.columns = ['date', 'ticker'] + needed
        self.long = long
        return self

    def clean(self, ref_col: str = 'Close',
              severe_threshold: float = 0.5) -> 'StockCleaner':
        long = self.long.sort_values(['ticker', 'date'])

        nan_frac = long.groupby('ticker')[ref_col].apply(lambda x: x.isna().mean())
        phantoms = nan_frac[nan_frac == 1.0].index.tolist()
        long = long[~long['ticker'].isin(phantoms)]

        nan_frac = long.groupby('ticker')[ref_col].apply(lambda x: x.isna().mean())
        severe = nan_frac[nan_frac > severe_threshold].index.tolist()
        long = long[~long['ticker'].isin(severe)]

        long[ref_col] = long.groupby('ticker')[ref_col].ffill()
        long = long.dropna(subset=[ref_col]).reset_index(drop=True)

        self.long = long
        return self

    def get(self) -> pd.DataFrame:
        return self.long


def align(*cleaners: StockCleaner) -> list[pd.DataFrame]:
    dfs = [c.get() for c in cleaners]
    key = ['date', 'ticker']
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=key, how='inner')
    cols = [key + [c for c in merged.columns if c not in key]]
    return merged
