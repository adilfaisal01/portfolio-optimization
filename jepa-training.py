from src.models.encoder import Encoder
from src.models.utils.mask_utils import apply_mask
from src.models.predictor import Predictor
import copy
import torch
from individual_stocks.data_class_parquet import StockMarketJEPADataset
from torch.utils.data import DataLoader
import os
from dataclasses import dataclass, field
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.models.utils.mask_utils import apply_mask


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
    batch_size: int = 1
    lr: float = 3e-4
    weight_decay: float = 0
    ema_momentum: float = 0.998
    num_epochs: int = 3
    model_path: str = "workspace/outputs"
    save_interval: int = 1

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
dataset=StockMarketJEPADataset(mask_ratio=jepa_setup.mask_ratio,num_patches=jepa_setup.num_patches,vix_fairweather=jepa_setup.vix_fairweather,parquet_path="individual_stocks/parquet_data/sector_etf_clean_trainingset.parquet")
data_loaded= DataLoader(dataset,batch_size=trainingsetup.batch_size,shuffle=True)

VAL_PARQUET_PATH = "individual_stocks/parquet_data/sector_etf_clean_testingset.parquet"
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

def save_model(model, epoch):
    save_path=trainingsetup.model_path+ "_epoch_" + str(epoch)+".pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        torch.save(model.state_dict(), save_path)
    except:
        print('lmao bruh, failure to save')

@torch.no_grad()
def evaluate(encoder, ema_encoder, predictor, loader):
    encoder.eval()
    predictor.eval()
    ema_encoder.eval()
    total = 0.0
    for window, masks, non_masks in loader:
        window = window.to(dev)
        masks = masks.to(dev)
        non_masks = non_masks.to(dev)
        targets = ema_encoder(window)
        targets = F.layer_norm(targets, (targets.size(-1),))
        targets = apply_mask(targets, masks)
        tokens = encoder(window, non_masks)
        pred = predictor(tokens, masks, non_masks)
        total += loss_pred(pred, targets).item()
    encoder.train()
    predictor.train()
    return total / len(loader)

def plot_loss_curve(train_losses, val_losses, val_epochs, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='train', linewidth=1.5)
    if val_losses:
        plt.plot(val_epochs, val_losses, 'ro-', label='val', markersize=6, linewidth=1.5)
    best_val = min(val_losses) if val_losses else None
    if best_val is not None:
        best_ep = val_epochs[val_losses.index(best_val)]
        plt.axhline(y=best_val, color='red', linestyle='--', alpha=0.5)
        plt.annotate(f'best val: {best_val:.4f} @ epoch {best_ep}',
                     xy=(best_ep, best_val), xytext=(best_ep + 0.5, best_val * 1.05),
                     fontsize=8, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('JEPA Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")

epoch_losses = []
val_losses = []
val_epochs = []

#EMA scheduling definition
ema_scheduler=(trainingsetup.ema_momentum+
    i*(1-trainingsetup.ema_momentum)/(trainingsetup.num_epochs)
    for i in range(int(trainingsetup.num_epochs+1)))

## trainingloop
output_dir = os.path.dirname(trainingsetup.model_path) or "."
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
    epoch_losses.append(avg_loss.item())

    if (epoch+1)%trainingsetup.save_interval==0:
        val_loss = evaluate(encoder, ema_encoder, predictor, val_loader)
        val_losses.append(val_loss)
        val_epochs.append(epoch + 1)
        save_model(encoder,(epoch+1))
        print(f"epoch {epoch+1}, train loss: {avg_loss:.4f}  |  val loss: {val_loss:.4f}  |  checkpoint saved")
        plot_loss_curve(epoch_losses, val_losses, val_epochs, output_dir)
    else:
        print(f"epoch {epoch+1}, train loss: {avg_loss:.4f}")

plot_loss_curve(epoch_losses, val_losses, val_epochs, output_dir)