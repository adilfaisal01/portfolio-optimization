import gymnasium as gym 
import numpy as np
import pandas as pd
from gymnasium import spaces

class Portfoliomarket(gym.Env):
    def __init__(self,n_assets:int, n_macro_indicators:int ,max_alloc:float, start_cap:float) -> None:
        self.n_assets=n_assets
        self.macro_indices=n_macro_indicators
        self.max_w=max_alloc
        self.start_cap=start_cap

    def _data_parser(self,dataset='sector_etf_clean_trainingset.parquet'):
        self.xlk_returns=dataset[dataset['ticker']=='xlk']['log_return']
        
    def step(self,u):
        pass

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def reward
        
