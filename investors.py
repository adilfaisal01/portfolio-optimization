import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rf_daily = 0.00012 #3% SOFR


for ticker in ['ARKK', 'BRK-B', 'SPY', 'PSHZF', 'BTC-USD']:

    wood = yf.Tickers(ticker).history(period='max', auto_adjust='True', start="2020-01-01", end="2025-12-31")['Close']
    
    wood['Return']= wood[ticker].pct_change()
    sharpe = (wood['Return'].mean() - rf_daily) / wood['Return'].std() * np.sqrt(252)
    print(f'{ticker} Sharpe: {sharpe:.2f}')
    
    years = (wood.index[-1] - wood.index[0]).days / 365.25
    
    total_return=wood[ticker].iloc[-1]/wood[ticker].iloc[0]
    cagr = (total_return)**(1/years) - 1
    
    print(f'{ticker} CAGR: {cagr:.2%}')
    
    wood_peak = wood[ticker].cummax()
    drawdown = (wood[ticker] - wood_peak) / wood_peak
    max_drawdown = abs(drawdown.min())
    print(f"{ticker} Max Drawdown: {max_drawdown:.2%}")

