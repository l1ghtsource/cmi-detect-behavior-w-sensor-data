import torch
import numpy as np

class MixupLoss:
    def __init__(self, criterion, num_classes, alpha=1.0):
        self.criterion = criterion
        self.num_classes = num_classes
        self.alpha = alpha
    
    def __call__(self, outputs, targets_a, targets_b, lam):
        return lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_batch(batch, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = batch['target'].size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_batch = {}
    for key in batch.keys():
        if key != 'target':
            mixed_batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]
        else:
            mixed_batch[key] = batch[key]
    
    targets_a = batch['target']
    targets_b = batch['target'][index]
    
    return mixed_batch, targets_a, targets_b, lam