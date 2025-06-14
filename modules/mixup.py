import torch
import numpy as np

class MixupLoss:
    def __init__(self, main_criterion, seq_type_criterion):
        self.main_criterion = main_criterion
        self.seq_type_criterion = seq_type_criterion
    
    def __call__(self, outputs, seq_type_outputs, targets_a, targets_b, seq_type_targets_a, seq_type_targets_b, lam):
        main_loss = lam * self.main_criterion(outputs, targets_a) + (1 - lam) * self.main_criterion(outputs, targets_b)
        seq_type_loss = lam * self.seq_type_criterion(seq_type_outputs, seq_type_targets_a) + (1 - lam) * self.seq_type_criterion(seq_type_outputs, seq_type_targets_b)
        return main_loss, seq_type_loss

def mixup_batch(batch, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = batch['main_target'].size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_batch = {}
    
    target_keys = {'main_target', 'seq_type_aux_target'}
    
    for key in batch.keys():
        if key not in target_keys:
            mixed_batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]
        else:
            mixed_batch[key] = batch[key]
    
    targets_a = batch['main_target']
    targets_b = batch['main_target'][index]
    
    seq_type_targets_a = batch['seq_type_aux_target']
    seq_type_targets_b = batch['seq_type_aux_target'][index]
    
    return mixed_batch, targets_a, targets_b, seq_type_targets_a, seq_type_targets_b, lam