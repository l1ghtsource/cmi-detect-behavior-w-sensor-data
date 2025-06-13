import random, os
import numpy as np
import torch
from configs.config import cfg

def seed_everything(seed: int = cfg.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def worker_init_fn(worker_id):
    seed = cfg.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)