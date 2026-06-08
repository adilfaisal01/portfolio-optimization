from src.models.encoder import Encoder
from src.models.utils.mask_utils import apply_mask
from src.models.utils
encoder_predictive= Encoder(dim_in=49,num_patches=21,kernel_size=49,embed_dim=256,embed_bias=True,nhead=8,jepa=True,num_layers=4)