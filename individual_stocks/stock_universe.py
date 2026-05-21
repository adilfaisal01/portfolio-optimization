import yfinance as yf
import pandas as pd

# collecting the data from wikipedia
# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
# data_sp500 = pd.read_html(
#     'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
#     storage_options=headers
# )[0]
# print(data_sp500)

# data_sp500.to_csv('S&P 500 data 2026.csv',index=False)

# making an index of the key stock matrices needed
# sp_tickers=pd.read_csv('S&P 500 data 2026.csv')['Symbol'].to_list()
# # # print(sp_tickers)

# # data=[]

# # for t in sp_tickers:
# #     information=yf.Ticker(t).info
# #     packet={
# #         'Ticker':t,
# #         'Market Cap':information.get('marketCap'),
# #         'Sector': information.get('sector')
# #     }
# #     data.append(packet)

# # df=pd.DataFrame(data)
# # df.to_csv('df.csv',index=False)

# pick the 20 biggest from my sector
df_sp500=pd.read_csv('df.csv')
# Your 11 sectors
your_sectors = [
    'Basic Materials', 'Communication Services',
    'Consumer Cyclical', 'Consumer Defensive',
    'Energy', 'Financial Services', 'Healthcare',
    'Industrials', 'Real Estate', 'Technology', 'Utilities'
]

universe = (
    df_sp500[df_sp500['Sector'].isin(your_sectors)]
    .groupby('Sector')
    .apply(lambda x: x.nlargest(20, 'Market Cap'))
    .reset_index(drop=True)
)

universe.to_csv('stock_universe_220.csv', index=False)
print(universe.groupby('Sector').size())
print(f"\nTotal stocks: {len(universe)}")
