import yfinance as yf
from datetime import datetime

assets = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'VNQ', 'VOX']
macro  = ['GLD', '^VIX', 'USO', 'IYR', 'SHV', 'TIP']
all_tickers = assets + macro

start = '2007-01-11'
end   = '2025-12-31'

print(f"Pulling {len(all_tickers)} tickers from {start} to {end}...")
data = yf.download(all_tickers, start=start, end=end, auto_adjust=True)
print(f"Shape: {data.shape}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"Columns (first 10): {data.columns.tolist()[:10]}")

out = 'sector_etf_data.parquet'
data.to_parquet(out)
print(f"Saved to {out}")
