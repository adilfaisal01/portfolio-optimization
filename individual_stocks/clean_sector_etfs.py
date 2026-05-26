"""Run StockCleaner on the sector ETF parquet with 2007-2025 range."""
import sys
sys.path.insert(0, '.')
from cleanup import StockCleaner, align
import pandas as pd
import numpy as np

PARQUET = 'sector_etf_data.parquet'
START = '2007-01-11'
END   = '2025-12-31'

print(f"=== Loading & cleaning sector ETFs ({START} to {END}) ===")

# Adjusted OHLC
ohlc = StockCleaner(PARQUET, start_date=START, end_date=END)
ohlc.extract(['Close', 'High', 'Low', 'Open']).clean()
print(f"OHLC tickers: {ohlc.get()['ticker'].nunique()}, rows: {len(ohlc.get()):,}")

# Unadjusted Volume
vol = StockCleaner(PARQUET, start_date=START, end_date=END)
vol.extract(['Volume']).clean()
print(f"Volume tickers: {vol.get()['ticker'].nunique()}, rows: {len(vol.get()):,}")

# Merge
df = pd.merge(ohlc.get(), vol.get()[['date', 'ticker', 'Volume']],
              on=['date', 'ticker'], how='inner')
df = df.rename(columns={'Volume': 'Volume_unadjusted'})
print(f"Merged shape: {df.shape}")

# ^VIX is an index with zero volume — set to 1 so RVOL = 1.0 (neutral)
mask_vix = df['ticker'] == '^VIX'
df.loc[mask_vix, 'Volume_unadjusted'] = 1.0
print(f"Fixed ^VIX volume: {mask_vix.sum()} rows set to 1.0")

# Feature engineering
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
df['log_return'] = df.groupby('ticker')['Close'].transform(
    lambda x: np.log(x.shift(-1) / x)
)
df['hl_spread'] = (df['High'] - df['Low']) / df['Open']
df['RVOL'] = df.groupby('ticker')['Volume_unadjusted'].transform(
    lambda x: x / (x.rolling(20, min_periods=1).mean())
)
df = df.dropna(subset=['log_return', 'hl_spread', 'RVOL']).reset_index(drop=True)
print(f"Final shape: {df.shape}")
print(f"Any remaining NaN: {df[['log_return', 'hl_spread', 'RVOL']].isna().any().any()}")
print(f"Tickers: {df['ticker'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

out = 'sector_etf_clean.parquet'
df.to_parquet(out)
print(f"Saved to {out}")
