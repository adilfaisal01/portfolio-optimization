import pandas as pd
import matplotlib.pyplot as plt

class DataExtractor:
    def __init__(self, ticker_list,macro_indices,dataset:str='sector_etf_clean_trainingset.parquet'):
        self.dataset_pframe:pd.DataFrame = pd.read_parquet(dataset)
        if macro_indices is None:
            self.macro_indices=['GLD','USO','IYR','SHV','TIP']
        else:
            self.macro_indices=macro_indices

        if ticker_list is None:
            self.ticker_list=['XLK','XLF','XLE','XLV','XLI','XLP','XLY','XLU','XLB','VNQ','VOX']
        else:
            self.ticker_list=ticker_list

    def get_macro(self):
        macro_data=self.dataset_pframe[self.dataset_pframe['ticker'].isin(self.macro_indices)]
        macro_logreturn=macro_data.pivot_table(index='date', columns='ticker',values='log_return')
        macro_spread=macro_data.pivot_table(index='date', columns='ticker', values='hl_spread')
        macro_rvol=macro_data.pivot_table(index='date',columns='ticker',values='RVOL')

        return macro_logreturn, macro_spread, macro_rvol

    def get_assets(self):
        etf_data=self.dataset_pframe[self.dataset_pframe['ticker'].isin(self.ticker_list)]
        etf_logreturn=etf_data.pivot_table(index='date', columns='ticker',values='log_return')
        etf_spread=etf_data.pivot_table(index='date', columns='ticker', values='hl_spread')
        etf_rvol=etf_data.pivot_table(index='date',columns='ticker',values='RVOL')
        return etf_logreturn, etf_spread, etf_rvol

    def get_vix(self,vix_fairweather):
        vix_data=self.dataset_pframe[self.dataset_pframe['ticker']=='^VIX']['Close']
        vix_normalized= vix_data/vix_fairweather
        return vix_normalized

