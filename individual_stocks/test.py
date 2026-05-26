import pandas as pd
import matplotlib.pyplot as plt

xy=pd.read_parquet("sector_etf_clean.parquet")

xli=xy[xy['ticker']=='^VIX']['Close']
xli_date=xy[xy['ticker']=='GLD']['date']
plt.plot(xli_date,xli)
plt.show()