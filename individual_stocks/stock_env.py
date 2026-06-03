import gymnasium as gym 
import numpy as np
import pandas as pd
from gymnasium import spaces
from dataextraction import DataExtractor

class Portfoliomarket(gym.Env):
    def __init__(self,n_assets:int, n_macro_indicators:int ,max_alloc:float, start_cap:float) -> None:
        self.n_assets=n_assets
        self.n_macro_indices=n_macro_indicators
        self.max_w=max_alloc
        self.start_cap=start_cap
        self.data=DataExtractor(ticker_list=None, macro_indices=None)
    def step(self,u):
        self.macro_lr, self.macro_sp,self.macro_rvol= self.data.get_macro()
        self.etf_lr, self.etf_sp, self.etf_rvol= self.data.get_assets()
        vix_data

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def reward(self):
        
