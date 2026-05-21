from tracemalloc import start
import pandas as pd
import numpy as np
import xarray as xr
class StockCleaner:
    def __init__(self, parquet_path: str,
                 start_date: str | None = None,
                 end_date: str | None = None):
        self.raw = pd.read_parquet(parquet_path)
        if start_date is not None:
            self.raw = self.raw.loc[self.raw.index >= start_date]
        if end_date is not None:
            self.raw = self.raw.loc[self.raw.index <= end_date]
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


def align(*cleaners: StockCleaner) -> pd.DataFrame:
    dfs = [c.get() for c in cleaners]
    key = ['date', 'ticker']
    result = dfs[0].copy()
    for df in dfs[1:]:
        extra_cols = [c for c in df.columns if c not in key and c not in result.columns]
        result = pd.merge(result, df[key + extra_cols], on=key, how='inner')
    return result


# --- Adjusted OHLC + unadjusted Volume pipeline ---
print("=== Loading & cleaning adjusted OHLC ===")
ohlc = StockCleaner('stock_info.parquet',start_date='2009-01-01',end_date='2020-12-31')
ohlc.extract(['Close', 'High', 'Low', 'Open']).clean()
print(f"OHLC tickers: {ohlc.get()['ticker'].nunique()}, rows: {len(ohlc.get()):,}")

print("\n=== Loading & cleaning unadjusted Volume ===")
vol = StockCleaner('stock_info_unadjusted.parquet',start_date='2009-01-01',end_date='2020-12-31')
vol.extract(['Volume']).clean()
print(f"Volume tickers: {vol.get()['ticker'].nunique()}, rows: {len(vol.get()):,}")

print("\n=== Merging on (date, ticker) ===")
df = pd.merge(ohlc.get(), vol.get()[['date', 'ticker', 'Volume']],
              on=['date', 'ticker'], how='inner')
df = df.rename(columns={'Volume': 'Volume_unadjusted'})
print(f"Merged shape: {df.shape}")
print(f"Tickers: {df['ticker'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
df['log_return'] = df.groupby('ticker')['Close'].transform(
lambda x: np.log(x.shift(-1) / x)
)
df['hl_spread'] =(df['High'] - df['Low'])/df['Open']
df = df.dropna(subset=['log_return', 'hl_spread']).reset_index(drop=True)
print(f"Final shape: {df.shape}")
print(f"Any remaining NaN: {df[['log_return', 'hl_spread', 'Volume_unadjusted']].isna().any().any()}")

print(df[df['ticker']=='PLTR'])

# print("\n=== Exporting to xarray ===")
# ds = df.set_index(['date', 'ticker']).to_xarray()
# ds = ds.fillna(0)
# ds.to_netcdf('price_data.nc')
# print("Saved to price_data.nc")

# print(f"\nTickers: {list(ds.ticker.values[:10])}... ({len(ds.ticker)} total)")
# print(f"Dates:   {ds.date.values[0]} to {ds.date.values[-1]} ({len(ds.date)} total)")
