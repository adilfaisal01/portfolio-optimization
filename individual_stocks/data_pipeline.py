import pandas as pd
import yfinance as yf

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

def DataProcessing(stock_data:pd.DataFrame,start_year:int,end_year:int,keyword:str):
    low_price_data=[]
    price_data=stock_data.loc[:,(keyword,slice(None))]
    for yy in range(start_year,end_year):
        yy2=str(yy)
        y1=price_data.loc[yy2]
        y1=y1[keyword]
        for ticker,ps in y1.items():
            for date,price in ps.items():
                low_price_data.append({'date':date, 'ticker': ticker, f'{keyword} price': price})
    return low_price_data

# pd.DataFrame(low_price_data).to_parquet('low_price.parquet')
# xpp_1= pd.read_parquet('low_price.parquet')
# print('1')
# print(xpp_1[xpp_1['ticker']=='AAPL'])
# 

word_list=['High','Close','Open','Low']

for word in word_list:
    price_list=DataProcessing(xx,2009,2021,word)
    pd.DataFrame(price_list).to_parquet(f'{word}_price.parquet')



