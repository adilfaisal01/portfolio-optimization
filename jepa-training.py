from src.models.encoder import Encoder
from src.models.utils.mask_utils import apply_mask
from src.models.predictor import Predictor
import copy
import torch
from individual_stocks.data_class_parquet import StockMarketJEPADataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
import torch.nn.functional as F
from src.models.utils.mask_utils import apply_mask

@dataclass
class Training_configuration:
    batch_size=1
    lr:float=3e-4
    weight_decay:float=0
    ema_momentum:float=0.998
    num_epochs:int=1
    model_path:str="workspace/outputs"
    save_interval:int=10

# @dataclass
# cl
    
trainingsetup=Training_configuration()
dev= torch.device("cuda" if torch.cuda.is_available() else "cpu")

## loading the dataset
dataset=StockMarketJEPADataset(mask_ratio=0.2,num_patches=20,vix_fairweather=20,parquet_path="individual_stocks/parquet_data/sector_etf_clean_trainingset.parquet")
data_loaded= DataLoader(dataset,batch_size=trainingsetup.batch_size,shuffle=True)

# print(len(data_loaded.dataset[0][0]))

# ## setting up the definitions of the NNs
encoder= Encoder(dim_in=49,num_patches=20,kernel_size=49,embed_dim=256,embed_bias=True,nhead=8,jepa=True,num_layers=4)
ema_encoder= copy.deepcopy(encoder)
predictor= Predictor(num_patches=20,num_layers=2,nhead=4,predictor_embed_dim=512,encoder_embed_dim=256)

## move all networks to the correct device
encoder.to(dev)
predictor.to(dev)
ema_encoder.to(dev)
# ema encoder is not differentiated
for p in ema_encoder.parameters():
    p.requires_grad=False

# define which parameters need to be optimized--> encoder and predictor
params_encoder=list(encoder.parameters())
params_predictor=list(predictor.parameters())
all_params_opt=params_encoder+params_predictor
optimizer= torch.optim.AdamW(all_params_opt, lr=trainingsetup.lr,weight_decay=trainingsetup.weight_decay)

def loss_pred(pred, target_ema):
    loss = 0.0
    for pred_i, target_ema_i in zip(pred, target_ema):
        loss = loss + torch.mean(torch.abs(pred_i - target_ema_i))
    loss /= len(pred)
    return loss

def save_model(model, epoch):
    save_path=trainingsetup.model_path+ "_epoch_" + str(epoch)+".pt"
    try:
        torch.save(model.state_dict(), save_path)
    except:
        print('lmao bruh, failure to save')
    

#EMA scheduling definition
ema_scheduler=(trainingsetup.ema_momentum+
    i*(1-trainingsetup.ema_momentum)/(trainingsetup.num_epochs)
    for i in range(int(trainingsetup.num_epochs+1)))

## trainingloop
for epoch in range(trainingsetup.num_epochs):
    m=next(ema_scheduler)
    total_loss=0.0
    for window, masks, non_masks in data_loaded:
        optimizer.zero_grad()
        window=window.to(dev)
        masks=masks.to(dev)
        non_masks=non_masks.to(dev)
        with torch.no_grad():
            target_values=ema_encoder(window)
            target_values=F.layer_norm(target_values, (target_values.size(-1),))
            target_values= apply_mask(target_values,masks)

        tokens=encoder(window,non_masks)
        pred=predictor(tokens,masks,non_masks)
        loss=loss_pred(pred,target_values)
        loss.backward()
        optimizer.step()

        # update the EMA encoder
        with torch.no_grad():
            for param_q, param_k in zip(
                encoder.parameters(), ema_encoder.parameters()
            ):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
            total_loss+=loss
    avg_loss=total_loss/len(data_loaded)
    print(f"epoch {epoch}, JEPA loss: {avg_loss: .4f}")

    if (epoch+1)%trainingsetup.save_interval==0:
        save_model(encoder,(epoch+1))