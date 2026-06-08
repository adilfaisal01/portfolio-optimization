from src.models.encoder import Encoder
from src.models.utils.mask_utils import apply_mask
from src.models.predictor import Predictor
import copy
import torch

dev= torch.device("cuda" if torch.cuda.is_available() else "cpu")
## setting up the definitions of the NNs
encoder_predictive= Encoder(dim_in=49,num_patches=21,kernel_size=49,embed_dim=256,embed_bias=True,nhead=8,jepa=True,num_layers=4)
ema_encoder= copy.deepcopy(encoder_predictive)
