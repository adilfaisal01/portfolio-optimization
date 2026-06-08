from torch.utils.data import Dataset
from dataextraction import DataExtractor
import torch
import random

class StockMarketJEPADataset(Dataset):
    def __init__(self,mask_ratio:float, num_patches:int, vix_fairweather:int,parquet_path:str='parquet_data/sector_etf_clean_trainingset.parquet'):
            self.num_patches=num_patches
            self.mask_ratio=mask_ratio
            self.parquet_path=parquet_path
            self.vix_fairweather=vix_fairweather
            data_extraction=DataExtractor(ticker_list=None,macro_indices=None,dataset=self.parquet_path)
            etf_lr, etf_sp, etf_rvol= data_extraction.get_assets()
            macro_lr,macro_sp,macro_rvol=data_extraction.get_macro()
            vix=data_extraction.get_vix(vix_fairweather=self.vix_fairweather)
            dat= torch.cat([
                torch.tensor(etf_lr.values, dtype=torch.float32),
                torch.tensor(etf_sp.values, dtype=torch.float32),
                torch.tensor(etf_rvol.values, dtype=torch.float32),
                torch.tensor(macro_lr.values, dtype=torch.float32),
                torch.tensor(macro_sp.values, dtype=torch.float32),
                torch.tensor(macro_rvol.values, dtype=torch.float32),
                torch.tensor(vix.values, dtype=torch.float32).unsqueeze(1),
            ],dim=1)
            self.data = dat[~torch.isnan(dat).any(dim=1)]
            self.num_windows = len(self.data)//self.num_patches
    def __len__(self)->int:
            return self.num_windows
    def __getitem__(self, index):
        start_idx= index*self.num_patches
        window= self.data[start_idx:start_idx+self.num_patches]
        num_masked = max(1, int(self.num_patches * self.mask_ratio))
        all_indices = list(range(self.num_patches))
        mask_idxs = sorted(random.sample(all_indices, num_masked))
        non_mask_idxs = [i for i in all_indices if i not in mask_idxs]

        return (
            window,
            torch.tensor(mask_idxs),
            torch.tensor(non_mask_idxs),
        )


if __name__ == "__main__":
    ds = StockMarketJEPADataset(num_patches=20, mask_ratio=0.15,vix_fairweather=20)
    patches, mask, non_mask = ds[0]
    print(f"patches shape:      {patches.shape}")  # [20, 49]
    print(f"mask indices:       {mask}")
    print(f"non-mask indices:   {non_mask.shape}")
    print(f"total windows:      {len(ds)}")

        
            
           
        
    
    