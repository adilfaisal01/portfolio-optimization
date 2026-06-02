import pandas as pd
import matplotlib.pyplot as plt

xx=pd.read_parquet('sector_etf_clean_testingset.parquet')
print(xx.columns)

class DataExtractor:
    def __init__(self, ticker,macro_indices,dataset='sector_etf_clean_trainingset.parquet'):
        self.dataset_pframe=pd.read_parquet(dataset)
        self.ticker=ticker
        if macro_indices is None:
            self.macro_indices=['GLD','USO','IYR','SHV','TIP']
        else:
            self.macro_indices=macro_indices

    def get_macro(self):
        macro_data=self.dataset_pframe[self.dataset_pframe['ticker'].isin(self.macro_indices)]
        macro_logreturn=macro_data.pivot_table(index='date', columns='ticker',values='log_return')
        macro_spread=macro_data.pivot_table(index='date', columns='ticker', values='hl_spread')
        
        

