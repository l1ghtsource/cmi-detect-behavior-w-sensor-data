import torch
import numpy as np

# class MixupLoss:
#     def __init__(self, criterion, num_classes, alpha=1.0):
#         self.criterion = criterion
#         self.num_classes = num_classes
#         self.alpha = alpha
    
#     def __call__(self, outputs, targets_a, targets_b, lam):
#         return lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)

# def mixup_data(x, y, alpha=1.0, device='cuda'):
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
    
#     batch_size = x.size()[0]
#     index = torch.randperm(batch_size).to(device)
    
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
    
#     return mixed_x, y_a, y_b, lam

class MixupLoss:
    def __init__(self, criterion, aux2_criterion, aux2_weight, num_classes, aux2_num_classes, alpha=1.0):
        self.criterion = criterion
        self.aux2_criterion = aux2_criterion
        self.aux2_weight = aux2_weight
        self.num_classes = num_classes
        self.aux2_num_classes = aux2_num_classes
        self.alpha = alpha
    
    def __call__(self, outputs, aux2_outputs, targets_a, targets_b, aux2_targets_a, aux2_targets_b, lam):
        main_loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
        aux2_loss = lam * self.aux2_criterion(aux2_outputs, aux2_targets_a) + (1 - lam) * self.aux2_criterion(aux2_outputs, aux2_targets_b)
        total_loss = main_loss + self.aux2_weight * aux2_loss
        return total_loss, main_loss, aux2_loss

# def mixup_batch(batch, alpha=1.0, device='cuda'):
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
    
#     batch_size = batch['target'].size()[0]
#     index = torch.randperm(batch_size).to(device)
    
#     mixed_batch = {}
#     for key in batch.keys():
#         if key != 'target':
#             mixed_batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]
#         else:
#             mixed_batch[key] = batch[key]
    
#     targets_a = batch['target']
#     targets_b = batch['target'][index]
    
#     return mixed_batch, targets_a, targets_b, lam

def mixup_batch(batch, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = batch['target'].size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_batch = {}
    
    target_keys = {'target', 'aux2_target'}
    
    for key in batch.keys():
        if key not in target_keys:
            mixed_batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]
        else:
            mixed_batch[key] = batch[key]
    
    targets_a = batch['target']
    targets_b = batch['target'][index]
    
    aux2_targets_a = batch['aux2_target']
    aux2_targets_b = batch['aux2_target'][index]
    
    return mixed_batch, targets_a, targets_b, aux2_targets_a, aux2_targets_b, lam