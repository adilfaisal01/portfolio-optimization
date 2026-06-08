from torch.utils.data import Dataset
from dataextraction import DataExtractor
import torch

class StockMarketJEPADataset(Dataset):
    def __init__(self,mask_ratio:float, num_patches:int, vix_fairweather:int,parquet_path:str='parquet_data/sector_etf_clean_trainingset.parquet'):
            self.num_patches=num_patches
            self.mask_ratio=mask_ratio
            self.parquet_path=parquet_path
            self.vix_fairweather=vix_fairweather
            data_extraction=DataExtractor(ticker_list=None,macro_indices=None,dataset=self.parquet_path)
            etf_lr, etf_sp, etf_rvol= data_extraction.get_assets()
            macro_lr,macro_sp,macro_rvol=data_extraction.get_macro()
            self.vix=data_extraction.get_vix(vix_fairweather=self.vix_fairweather)

    def gg(self):
        return self.vix

if __name__=="__main__":
    ds=StockMarketJEPADataset(mask_ratio=0.3, num_patches=20, vix_fairweather=20)
    print(type(ds.vix))