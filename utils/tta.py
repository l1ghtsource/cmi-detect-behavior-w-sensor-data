import torch
from data.ts_augmentations import jitter, magnitude_warp, time_warp, scaling

def apply_tta(batch, tta_strategies):
    augmented_batches = [batch] # original batch
    
    for strategy_name, strategy_params in tta_strategies.items():
        aug_batch = {}
        
        for key, tensor in batch.items():
            if key in ['target', 'aux_target']:
                aug_batch[key] = tensor
                continue
                
            if key == 'demographics':
                aug_batch[key] = tensor
                continue
                
            aug_data = tensor.clone()
            
            if strategy_name == 'jitter' and key in strategy_params.get('sensors', ['imu', 'thm', 'tof']):
                sigma = strategy_params.get('sigma', 0.03)
                for i in range(aug_data.shape[0]):
                    aug_data[i] = torch.tensor(jitter(aug_data[i].numpy(), sigma=sigma), dtype=torch.float32)
                    
            elif strategy_name == 'magnitude_warp' and key in strategy_params.get('sensors', ['imu', 'thm']):
                sigma = strategy_params.get('sigma', 0.15)
                knot = strategy_params.get('knot', 3)
                for i in range(aug_data.shape[0]):
                    aug_data[i] = torch.tensor(magnitude_warp(aug_data[i].numpy(), sigma=sigma, knot=knot), dtype=torch.float32)
                    
            elif strategy_name == 'time_warp' and key in strategy_params.get('sensors', ['imu', 'thm', 'tof']):
                sigma = strategy_params.get('sigma', 0.1)
                knot = strategy_params.get('knot', 3)
                for i in range(aug_data.shape[0]):
                    aug_data[i] = torch.tensor(time_warp(aug_data[i].numpy(), sigma=sigma, knot=knot), dtype=torch.float32)
                    
            elif strategy_name == 'scaling' and key in strategy_params.get('sensors', ['imu', 'thm']):
                sigma = strategy_params.get('sigma', 0.08)
                for i in range(aug_data.shape[0]):
                    aug_data[i] = torch.tensor(scaling(aug_data[i].numpy(), sigma=sigma), dtype=torch.float32)
            
            aug_batch[key] = aug_data
            
        augmented_batches.append(aug_batch)
    
    return augmented_batches