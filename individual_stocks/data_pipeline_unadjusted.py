import pandas as pd
import yfinance as yf

# collecting stock data

def yahoo_data_pipeline(start_date:str,end_date:str,autoadjustment:bool,parquetname:str):

    df=pd.read_csv('stock_universe_220.csv')['Ticker'].to_list()
    tickers = yf.Tickers(df)
    stock_info=yf.download(df,start=start_date,end=end_date,auto_adjust=autoadjustment)
    stock_info.to_parquet(f'{parquetname}.parquet')
    return None

# yahoo_data_pipeline(start_date='2009-01-01',end_date='2025-12-31',autoadjustment=False,parquetname='stock_info_unadjusted')

# stock_info.loc[:, ('Close', slice(None))]        # Close for all tickers

xx_unadjusted=pd.read_parquet('stock_info_unadjusted.parquet')
print(xx_unadjusted)

