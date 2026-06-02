import gymnasium as gym 
import numpy as np

class Portfoliomarket(gym.Env):
    def __init__(self,n_assets:int, n_macro_indicators:int ,max_alloc:) -> None:
        
