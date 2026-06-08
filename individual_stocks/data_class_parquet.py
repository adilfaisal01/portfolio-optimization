from torch.utils.data import Dataset
from dataextraction import DataExtractor
class StockMarketDataset(Dataset):
    def __init__(
        self,
        parquet_file_path:str,
        mask_ratio:float=0.7,
        window_size:int=21,

        ):
            self.window_size=window_size
            self.mask_ratio=mask_ratio
            extractor=data
            
        