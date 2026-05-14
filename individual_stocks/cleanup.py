import pandas as pd
import numpy as np
import xarray as xr

price_cats = ['High', 'Low', 'Close', 'Open']

print("=== Layer 0: Merge OHLC parquets ===")
merged = None
for cat in price_cats:
    df = pd.read_parquet(f'{cat}_price.parquet')
    if merged is None:
        merged = df
    else:
        merged = pd.merge(merged, df, on=['date', 'ticker'])
merged = merged.sort_values(['ticker', 'date']).reset_index(drop=True)
print(f"Merged shape: {merged.shape}")
print(f"Tickers before cleanup: {merged['ticker'].nunique()}")

print("\n=== Layer 1: Drop phantom tickers (100% NaN Close price) ===")
nan_by_ticker = merged.groupby('ticker')['Close price'].apply(lambda x: x.isna().mean())
phantoms = nan_by_ticker[nan_by_ticker == 1.0].index.tolist()
print(f"Phantoms: {phantoms}")
merged = merged[~merged['ticker'].isin(phantoms)]
print(f"Tickers after phantom removal: {merged['ticker'].nunique()}")

print("\n=== Layer 2: Drop tickers with >50% NaN Close price ===")
nan_by_ticker = merged.groupby('ticker')['Close price'].apply(lambda x: x.isna().mean())
severe_drop = nan_by_ticker[nan_by_ticker > 0.50].index.tolist()
print(f"Severe drop (>50%% NaN): {severe_drop}")
merged = merged[~merged['ticker'].isin(severe_drop)]
print(f"Tickers after >50%% NaN drop: {merged['ticker'].nunique()}")

print("\n=== Layer 3: Forward-fill Close price gaps within each ticker ===")
merged = merged.sort_values(['ticker', 'date'])
merged['Close price'] = merged.groupby('ticker')['Close price'].ffill()
remaining_nan_close = merged['Close price'].isna().sum()
print(f"Remaining NaN in Close price: {remaining_nan_close}")

print("\n=== Layer 4: Compute log_return = log(P_{t+1} / P_t) ===")
merged['log_return'] = merged.groupby('ticker')['Close price'].transform(
    lambda x: np.log(x.shift(-1) / x)
)

print("\n=== Layer 5: Compute hl_spread ===")
merged['hl_spread'] = merged['High price'] - merged['Low price']

print("\n=== Layer 6: Final dropna (date, how='any') for dense matrix ===")
merged = merged.sort_values(['date', 'ticker'])
merged = merged.dropna(subset=['log_return', 'hl_spread']).reset_index(drop=True)
print(f"Final shape: {merged.shape}")
print(f"Final ticker count: {merged['ticker'].nunique()}")
print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
print(f"Any remaining NaN: {merged[['log_return', 'hl_spread']].isna().any().any()}")

print("\n=== Layer 7: Build xarray Dataset ===")
ds = merged.set_index(['date', 'ticker']).to_xarray()
ds = ds.fillna(0)
ds.to_netcdf('price_data.nc')
print("Saved to price_data.nc")

print("\n=== Layer 8: Preview ===")
print(ds)
print(f"\nTickers: {list(ds.ticker.values[:10])}... ({len(ds.ticker)} total)")
print(f"Dates:   {ds.date.values[0]} to {ds.date.values[-1]} ({len(ds.date)} total)")
print(f"\nSample log_return (first 3 AAPL rows):")
print(ds['log_return'].sel(ticker='AAPL').head(3).values)
