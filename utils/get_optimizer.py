from torch.optim import AdamW
from optimizers.adan import Adan
from optimizers.adamp import AdamP
from optimizers.madgrad import MADGRAD
from configs.config import cfg

def get_optimizer(params):
    if cfg.optim_type == 'adamw':
        return AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim_type == 'adan':
        return Adan(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim_type == 'adamp':
        return AdamP(params, lr=cfg.lr, weight_decay=cfg.weight_decay) 
    elif cfg.optim_type == 'madgrad':
        return MADGRAD(params, lr=cfg.lr, weight_decay=cfg.weight_decay) 
    else:
        raise Exception('stick your finger in your ass !!')