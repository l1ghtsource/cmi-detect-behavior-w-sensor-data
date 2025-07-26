import torch
import numpy as np

class MixupLoss:
    def __init__(self, main_criterion, hybrid_criterions, seq_type_criterion, orientation_criterion):
        self.main_criterion = main_criterion
        self.hybrid_criterions = hybrid_criterions
        self.seq_type_criterion = seq_type_criterion
        self.orientation_criterion = orientation_criterion
    
    def __call__(self, outputs, seq_type_outputs, orientation_outputs, ext1_out1, ext2_out1, ext3_out1, ext4_out1, targets_a, targets_b, seq_type_targets_a, seq_type_targets_b, orientation_targets_a, orientation_targets_b, lam):
        main_loss = lam * self.main_criterion(outputs, targets_a) + (1 - lam) * self.main_criterion(outputs, targets_b)
        ext1_out1_loss = lam * self.hybrid_criterions[0](ext1_out1, targets_a) + (1 - lam) * self.hybrid_criterions[0](ext1_out1, targets_b)
        ext2_out1_loss = lam * self.hybrid_criterions[1](ext2_out1, targets_a) + (1 - lam) * self.hybrid_criterions[1](ext2_out1, targets_b)
        ext3_out1_loss = lam * self.hybrid_criterions[2](ext3_out1, targets_a) + (1 - lam) * self.hybrid_criterions[2](ext3_out1, targets_b) 
        ext4_out1_loss = lam * self.hybrid_criterions[3](ext4_out1, targets_a) + (1 - lam) * self.hybrid_criterions[3](ext4_out1, targets_b)
        seq_type_loss = lam * self.seq_type_criterion(seq_type_outputs, seq_type_targets_a) + (1 - lam) * self.seq_type_criterion(seq_type_outputs, seq_type_targets_b)
        orientation_loss = lam * self.orientation_criterion(orientation_outputs, orientation_targets_a) + (1 - lam) * self.orientation_criterion(orientation_outputs, orientation_targets_b)
        return main_loss, ext1_out1_loss, ext2_out1_loss, ext3_out1_loss, ext4_out1_loss, seq_type_loss, orientation_loss

def mixup_batch(batch, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = batch['main_target'].size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_batch = {}
    
    target_keys = {'main_target', 'seq_type_aux_target', 'orientation_aux_target'}
    
    for key in batch.keys():
        if key not in target_keys:
            mixed_batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]
        else:
            mixed_batch[key] = batch[key]
    
    targets_a = batch['main_target']
    targets_b = batch['main_target'][index]
    
    seq_type_targets_a = batch['seq_type_aux_target']
    seq_type_targets_b = batch['seq_type_aux_target'][index]

    orientation_targets_a = batch['orientation_aux_target']
    orientation_targets_b = batch['orientation_aux_target'][index]
    
    return mixed_batch, targets_a, targets_b, seq_type_targets_a, seq_type_targets_b, orientation_targets_a, orientation_targets_b, lam

# mixup only in (label <= 7) or (label > 7) groups
# def mixup_batch(batch, alpha=1.0, device='cuda'):
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1

#     main_targets = batch['main_target']
#     batch_size = main_targets.size(0)

#     cat1_idx = torch.where(main_targets <= 7)[0]
#     cat2_idx = torch.where(main_targets > 7)[0]

#     perm = torch.arange(batch_size, device=main_targets.device)

#     if len(cat1_idx) > 1:
#         perm[cat1_idx] = cat1_idx[torch.randperm(len(cat1_idx), device=device)]
#     if len(cat2_idx) > 1:
#         perm[cat2_idx] = cat2_idx[torch.randperm(len(cat2_idx), device=device)]

#     mixed_batch = {}
    
#     target_keys = {'main_target', 'seq_type_aux_target'}

#     for key, tensor in batch.items():
#         if key not in target_keys:
#             mixed_batch[key] = lam * tensor + (1.0 - lam) * tensor[perm]
#         else:
#             mixed_batch[key] = tensor

#     targets_a = batch['main_target']
#     targets_b = batch['main_target'][perm]
#     seq_type_targets_a = batch['seq_type_aux_target']
#     seq_type_targets_b = batch['seq_type_aux_target'][perm]

#     return mixed_batch, targets_a, targets_b, seq_type_targets_a, seq_type_targets_b, lam