import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
#collecting stock data

# df=pd.read_csv('stock_universe_220.csv')['Ticker'].to_list()

# tickers = yf.Tickers(df)
# stock_info=yf.download(df,start='2009-01-01',end='2025-12-31',auto_adjust=True)
# stock_info.to_parquet('stock_info.parquet')

# reading the parquet data
xx=pd.read_parquet('stock_info.parquet')
# reading data
# But heads up — for your MultiIndex parquet data, positional indexing is fragile since column order might shift. Named indexing is safer with that structure:
# stock_info[('Close', 'NVDA')]                    # single column → Series
# stock_info.xs('NVDA', level=1, axis=1)           # all features for one ticker
# stock_info.loc[:, ('Close', slice(None))]        # Close for all tickers


## find the slice of trades done by a stock on a calendar year
# allows for good data to be added to the 
volume_data=xx.loc[:,('Volume',slice(None))]
volumes_share=[]
for y in range(2009,2021):
    yy=str(y)
    yearly_volume=volume_data.loc[yy] ## get the volume data for the full year
    list_volume_year=yearly_volume.sum().droplevel(0)
    stock_volume_share=list_volume_year/(list_volume_year.sum())
    for ticker, val in stock_volume_share.items():
            volumes_share.append({'year': y, 'ticker': ticker, 'vol_share': val})

    

# pd.DataFrame(volumes_share).to_parquet('volume_share.parquet')

# xpp= pd.read_parquet('volume_share.parquet')
# print(xpp['year'])

## get the 