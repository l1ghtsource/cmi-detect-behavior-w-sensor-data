import torch
from data.ts_augmentations import jitter, magnitude_warp, time_warp, scaling

def apply_tta(batch, tta_strategies):
    augmented_batches = [batch] # original batch

    for strategy_name, strategy_params in tta_strategies.items():
        aug_batch = {}

        for key, tensor in batch.items():
            if key in ['target', 'aux_target', 'demographics']:
                aug_batch[key] = tensor
                continue

            aug_data = tensor.clone()
            
            if key not in strategy_params.get('sensors', ['imu', 'thm', 'tof']):
                aug_batch[key] = aug_data
                continue

            if strategy_name == 'jitter':
                sigma = strategy_params.get('sigma', 0.03)
                aug_func = lambda x: jitter(x, sigma=sigma)
            elif strategy_name == 'magnitude_warp':
                sigma = strategy_params.get('sigma', 0.15)
                knot = strategy_params.get('knot', 3)
                aug_func = lambda x: magnitude_warp(x, sigma=sigma, knot=knot)
            elif strategy_name == 'time_warp':
                sigma = strategy_params.get('sigma', 0.1)
                knot = strategy_params.get('knot', 3)
                aug_func = lambda x: time_warp(x, sigma=sigma, knot=knot)
            elif strategy_name == 'scaling':
                sigma = strategy_params.get('sigma', 0.08)
                aug_func = lambda x: scaling(x, sigma=sigma)
            else:
                aug_batch[key] = aug_data
                continue

            # aug_data: (batch_size, n_sensors, seq_len, n_vars)
            batch_size, n_sensors, seq_len, n_vars = aug_data.shape
            
            for batch_idx in range(batch_size):
                for sensor_idx in range(n_sensors):
                    # (seq_len, n_vars)
                    sensor_data = aug_data[batch_idx, sensor_idx].numpy()
                    augmented_sensor_data = aug_func(sensor_data)
                    aug_data[batch_idx, sensor_idx] = torch.tensor(
                        augmented_sensor_data, dtype=torch.float32
                    )
            aug_batch[key] = aug_data

        augmented_batches.append(aug_batch)

    return augmented_batches
