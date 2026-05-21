import pandas as pd
import yfinance as yf

# collecting stock data

def yahoo_data_pipeline(start_date:str,end_date:str,autoadjustment:bool,parquetname:str):

    df=pd.read_csv('stock_universe_220.csv')['Ticker'].to_list()
    tickers = yf.Tickers(df)
    stock_info=yf.download(df,start=start_date,end=end_date,auto_adjust=autoadjustment)
    stock_info.to_parquet(f'{parquetname}.parquet')
    return None


# reading the parquet data

# reading data
# But heads up — for your MultiIndex parquet data, positional indexing is fragile since column order might shift. Named indexing is safer with that structure:
# stock_info[('Close', 'NVDA')]                    # single column → Series
# stock_info.xs('NVDA', level=1, axis=1)           # all features for one ticker
# stock_info.loc[:, ('Close', slice(None))]        # Close for all tickers


## find the slice of trades done by a stock on a calendar year
# allows for good data to be added to the 


