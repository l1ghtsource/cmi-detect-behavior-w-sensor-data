import torch
import numpy as np

class MixupLoss:
    def __init__(self, main_criterion, hybrid_criterions, seq_type_criterion, orientation_criterion):
        self.main_criterion = main_criterion
        self.hybrid_criterions = hybrid_criterions
        self.seq_type_criterion = seq_type_criterion
        self.orientation_criterion = orientation_criterion
    
    def __call__(self, outputs, seq_type_outputs, orientation_outputs, 
                 ext1_out1, ext2_out1, ext3_out1, ext4_out1, 
                 targets_list, seq_type_targets_list, orientation_targets_list, n):
        
        main_loss = 0
        ext1_out1_loss = 0
        ext2_out1_loss = 0
        ext3_out1_loss = 0
        ext4_out1_loss = 0
        seq_type_loss = 0
        orientation_loss = 0
        
        lam = 1.0 / n
        
        for i in range(n):
            main_loss += lam * self.main_criterion(outputs, targets_list[i])
            ext1_out1_loss += lam * self.hybrid_criterions[0](ext1_out1, targets_list[i])
            ext2_out1_loss += lam * self.hybrid_criterions[1](ext2_out1, targets_list[i])
            ext3_out1_loss += lam * self.hybrid_criterions[2](ext3_out1, targets_list[i])
            ext4_out1_loss += lam * self.hybrid_criterions[3](ext4_out1, targets_list[i])
            seq_type_loss += lam * self.seq_type_criterion(seq_type_outputs, seq_type_targets_list[i])
            orientation_loss += lam * self.orientation_criterion(orientation_outputs, orientation_targets_list[i])
        
        return main_loss, ext1_out1_loss, ext2_out1_loss, ext3_out1_loss, ext4_out1_loss, seq_type_loss, orientation_loss
    
def mixup_batch(batch, alpha=1.0, device='cuda'):
    batch_size = batch['main_target'].size(0)
    
    n = np.random.choice([2, 3, 4, 5])
    
    mixed_batch = {}
    target_keys = {'main_target', 'seq_type_aux_target', 'orientation_aux_target'}
    
    indices = []
    for i in range(n):
        perm = torch.randperm(batch_size).to(device)
        indices.append(perm)
    
    for key in batch.keys():
        if key not in target_keys:
            original_data = batch[key]
            
            if len(original_data.shape) == 4:  # (B, sensors, seq_len, channels)
                seq_len = original_data.size(2)
                segment_len = seq_len // n
                mixed_data = original_data.clone()
                
                for i in range(n):
                    start = i * segment_len
                    if i == n - 1:
                        end = seq_len
                    else:
                        end = start + segment_len
                    mixed_data[:, :, start:end, :] = original_data[indices[i]][:, :, start:end, :]
                
                mixed_batch[key] = mixed_data
                
            elif len(original_data.shape) == 3:  # (B, seq_len, features)
                seq_len = original_data.size(1)
                segment_len = seq_len // n
                mixed_data = original_data.clone()
                
                for i in range(n):
                    start = i * segment_len
                    if i == n - 1:
                        end = seq_len
                    else:
                        end = start + segment_len
                    mixed_data[:, start:end, :] = original_data[indices[i]][:, start:end, :]
                
                mixed_batch[key] = mixed_data
                
            elif len(original_data.shape) == 2:  # (B, features)
                feat_len = original_data.size(1)
                segment_len = feat_len // n
                mixed_data = original_data.clone()
                
                for i in range(n):
                    start = i * segment_len
                    if i == n - 1:
                        end = feat_len
                    else:
                        end = start + segment_len
                    mixed_data[:, start:end] = original_data[indices[i]][:, start:end]
                
                mixed_batch[key] = mixed_data
            else:
                mixed_batch[key] = original_data
        else:
            mixed_batch[key] = batch[key]
    
    targets_list = [batch['main_target'][indices[i]] for i in range(n)]
    seq_type_targets_list = [batch['seq_type_aux_target'][indices[i]] for i in range(n)]
    orientation_targets_list = [batch['orientation_aux_target'][indices[i]] for i in range(n)]
    
    return mixed_batch, targets_list, seq_type_targets_list, orientation_targets_list, n