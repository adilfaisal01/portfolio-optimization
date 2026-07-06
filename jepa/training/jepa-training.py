from jepa.models.encoder import Encoder
from jepa.models.utils.mask_utils import apply_mask
from jepa.models.predictor import Predictor
import copy
import torch
from jepa.data.data_class_parquet import StockMarketJEPADataset
from torch.utils.data import DataLoader
import os
from dataclasses import dataclass, field
import torch.nn.functional as F
import matplotlib.pyplot as plt
from jepa.models.utils.mask_utils import apply_mask


def _init_from_env(obj, prefix: str) -> None:
    for field_name in obj.__dataclass_fields__:
        env_val = os.environ.get(f"{prefix}{field_name.upper()}")
        if env_val is not None:
            field_type = type(getattr(obj, field_name))
            try:
                setattr(obj, field_name, field_type(env_val))
            except (ValueError, TypeError):
                raise ValueError(
                    f"Cannot convert env var {prefix}{field_name.upper()}={env_val!r} "
                    f"to {field_type.__name__}"
                )


@dataclass
class Training_configuration:
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 0
    ema_momentum: float = 0.998
    num_epochs: int = 3
    model_path: str = "workspace/outputs"
    save_interval: int = 1
    lambda_v:float=0.5
    lambda_cv:float=0.45

    def __post_init__(self):
        _init_from_env(self, "TRAIN_")


@dataclass
class JEPA_parameters:
    mask_ratio: float = 0.2
    num_patches: int = 20
    vix_fairweather: int = 20
    predictor_embed_dim: int = 512
    encoder_embed_dim: int = 256
    kernel_size: int = 49
    dim_in_encoder: int = 49
    num_layers_encoder: int = 4
    num_layers_predictor: int = 2
    nhead_encoder: int = 8
    n_head_predictor: int = 2

    def __post_init__(self):
        _init_from_env(self, "JEPA_")

jepa_setup=JEPA_parameters()    
trainingsetup=Training_configuration()
dev= torch.device("cuda" if torch.cuda.is_available() else "cpu")

## loading the training dataset
dataset=StockMarketJEPADataset(mask_ratio=jepa_setup.mask_ratio,num_patches=jepa_setup.num_patches,vix_fairweather=jepa_setup.vix_fairweather,parquet_path="jepa/data/parquet_data/sector_etf_clean_trainingset.parquet")
data_loaded= DataLoader(dataset,batch_size=trainingsetup.batch_size,shuffle=True)

VAL_PARQUET_PATH = "jepa/data/parquet_data/sector_etf_clean_testingset.parquet"
val_dataset = StockMarketJEPADataset(mask_ratio=jepa_setup.mask_ratio, num_patches=jepa_setup.num_patches, vix_fairweather=jepa_setup.vix_fairweather, parquet_path=VAL_PARQUET_PATH)
val_loader = DataLoader(val_dataset, batch_size=trainingsetup.batch_size, shuffle=False)

# print(len(data_loaded.dataset[0][0]))

# ## setting up the definitions of the NNs
encoder= Encoder(dim_in=jepa_setup.dim_in_encoder,
    num_patches=jepa_setup.num_patches,
    kernel_size=jepa_setup.kernel_size,
    embed_dim=jepa_setup.encoder_embed_dim,
    embed_bias=True,nhead=jepa_setup.nhead_encoder,
    jepa=True,
    num_layers=jepa_setup.num_layers_encoder)
ema_encoder= copy.deepcopy(encoder)
predictor= Predictor(num_patches=20,
    num_layers=jepa_setup.num_layers_predictor,
    nhead=jepa_setup.n_head_predictor,
    predictor_embed_dim=jepa_setup.predictor_embed_dim,
    encoder_embed_dim=jepa_setup.encoder_embed_dim)

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

def save_model(encoder, predictor, epoch):
    base_path = trainingsetup.model_path
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    try:
        torch.save(encoder.state_dict(), f"{base_path}_model_epoch_{epoch}_encoder.pt")
        torch.save(predictor.state_dict(), f"{base_path}_model_epoch_{epoch}_predictor.pt")
    except:
        print('lmao bruh, failure to save')


def vicreg_eval(z:torch.Tensor, gamma=1.0, epsilon:float=1e-5):
     N,D= z.shape #N: batch_size, D: dimension of the representations
     # variance --> V
     std= torch.sqrt(z.var(dim=0)+epsilon)
     variance_loss= torch.mean(torch.relu(gamma-std))
     #covariance --> C
     center_z= z-z.mean(dim=0)
     covar= (center_z.T@center_z)/(N-1)
     diagonal_covar= covar*torch.eye(D,device=covar.device)
     off_diag_covar= covar-diagonal_covar
     covar_loss=(off_diag_covar**2).sum()/D
     # SVD breakdown
     u,s,vh= torch.linalg.svd(center_z, full_matrices= False)
     p= s/s.sum()
     effective_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-10)))

     ## max auto corr (off diagonal)    
     corr=covar/(torch.outer(std,std)+epsilon)
     max_corr = corr[~torch.eye(D, dtype=torch.bool)].abs().max()

     return {
             'variance_loss': variance_loss,
             'covariance_loss': covar_loss,
             'mean_std': std.mean().item(),
             'min_std': std.min().item(),
             'effective_rank': effective_rank.item(),
             'max_corr': max_corr.item(),

      }


@torch.no_grad()
def evaluate(encoder, ema_encoder, predictor, loader):
    encoder.eval()
    predictor.eval()
    ema_encoder.eval()
    total = 0.0
    pred_loss_sum = 0.0
    var_loss_sum = 0.0
    cov_loss_sum = 0.0
    for window, masks, non_masks in loader:
        window = window.to(dev)
        masks = masks.to(dev)
        non_masks = non_masks.to(dev)
        targets = ema_encoder(window)
        targets = F.layer_norm(targets, (targets.size(-1),))
        # targets= F.normalize(targets, dim=-1)
        targets = apply_mask(targets, masks)
        tokens = encoder(window, non_masks)
        pred = predictor(tokens, masks, non_masks)

        pred_loss = loss_pred(pred, targets)
        vic_eval=vicreg_eval(tokens.view(-1, tokens.size(-1)))
        total += (pred_loss + trainingsetup.lambda_v*vic_eval['variance_loss'] + trainingsetup.lambda_cv*vic_eval['covariance_loss']).item()
        pred_loss_sum += pred_loss.item()
        var_loss_sum += vic_eval['variance_loss'].item()
        cov_loss_sum += vic_eval['covariance_loss'].item()
    encoder.train()
    predictor.train()
    n = len(loader)
    return total / n, pred_loss_sum / n, var_loss_sum / n, cov_loss_sum / n

def plot_all_curves(train_total, val_total, train_pred, val_pred, train_var, val_var, train_cov, val_cov, val_epochs, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(train_total) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, train_total, label='train', linewidth=1.5)
    if val_total:
        ax.plot(val_epochs, val_total, 'ro-', label='val', markersize=6, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(epochs, train_pred, label='train', linewidth=1.5)
    if val_pred:
        ax.plot(val_epochs, val_pred, 'ro-', label='val', markersize=6, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Prediction Loss')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(epochs, train_var, label='train', linewidth=1.5)
    if val_var:
        ax.plot(val_epochs, val_var, 'ro-', label='val', markersize=6, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Variance Loss')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(epochs, train_cov, label='train', linewidth=1.5)
    if val_cov:
        ax.plot(val_epochs, val_cov, 'ro-', label='val', markersize=6, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Covariance Loss')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'component_loss_curve.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Component loss curves saved to {save_path}")

epoch_losses = []
val_losses = []
val_epochs = []
# New lists for component losses
train_pred_losses = []
train_var_losses = []
train_cov_losses = []
val_pred_losses = []
val_var_losses = []
val_cov_losses = []

#EMA scheduling definition
ema_scheduler=(trainingsetup.ema_momentum+
    i*(1-trainingsetup.ema_momentum)/(trainingsetup.num_epochs)
    for i in range(int(trainingsetup.num_epochs+1)))

## trainingloop
output_dir = os.path.dirname(trainingsetup.model_path) or "."
for epoch in range(trainingsetup.num_epochs):
    m=next(ema_scheduler)
    total_loss=0.0
    pred_loss_sum = 0.0
    var_loss_sum = 0.0
    cov_loss_sum = 0.0
    for window, masks, non_masks in data_loaded:
        optimizer.zero_grad()
        window=window.to(dev)
        masks=masks.to(dev)
        non_masks=non_masks.to(dev)
        # print(f'shape of data being input: {window.shape}')
       
        with torch.no_grad():
            target_values=ema_encoder(window)
            target_values=F.layer_norm(target_values, (target_values.size(-1),))
            #target_values=F.normalize(target_values,dim=-1)
            target_values= apply_mask(target_values,masks)

        tokens=encoder(window,non_masks)
        pred=predictor(tokens,masks,non_masks)
        # compute individual loss components
        pred_loss = loss_pred(pred,target_values)
        vic=vicreg_eval(tokens.view(-1, tokens.size(-1)))
        loss = pred_loss + trainingsetup.lambda_v*vic['variance_loss'] + trainingsetup.lambda_cv*vic['covariance_loss']
        # accumulate component losses for training
        pred_loss_sum += pred_loss.item()
        var_loss_sum += vic['variance_loss'].item()
        cov_loss_sum += vic['covariance_loss'].item()
        # print(f'token size: {pred.shape}')
        # print(f'target size: {target_values.shape}')
        # 
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
    epoch_losses.append(avg_loss.item())
    train_pred_losses.append(pred_loss_sum / len(data_loaded))
    train_var_losses.append(var_loss_sum / len(data_loaded))
    train_cov_losses.append(cov_loss_sum / len(data_loaded))

    if (epoch+1)%trainingsetup.save_interval==0:
        val_loss, val_pred, val_var, val_cov = evaluate(encoder, ema_encoder, predictor, val_loader)
        val_losses.append(val_loss)
        val_pred_losses.append(val_pred)
        val_var_losses.append(val_var)
        val_cov_losses.append(val_cov)
        val_epochs.append(epoch + 1)
        save_model(encoder, predictor, (epoch+1))
        print(f"epoch {epoch+1}, train loss: {avg_loss:.4f}  |  val loss: {val_loss:.4f}  |  checkpoint saved")
        # after epoch finishes, plot all curves
        plot_all_curves(epoch_losses, val_losses, train_pred_losses, val_pred_losses, train_var_losses, val_var_losses, train_cov_losses, val_cov_losses, val_epochs, output_dir)
    else:
        print(f"epoch {epoch+1}, train loss: {avg_loss:.4f}")

plot_all_curves(epoch_losses, val_losses, train_pred_losses, val_pred_losses, train_var_losses, val_var_losses, train_cov_losses, val_cov_losses, val_epochs, output_dir)