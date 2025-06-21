import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from data.ts_augmentations import jitter, magnitude_warp, time_warp, scaling

def apply_rotation_augmentation_numpy(data, max_angle=15):
    if data.shape[1] != 7:
        return data
    
    acc_data = data[:, :3]  # (seq_len, 3)
    quat_data = data[:, 3:]  # (seq_len, 4): [w, x, y, z]
    
    if max_angle > 0:
        angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
        random_rotation = R.from_euler('z', angle)
    else:
        random_rotation = R.random()
    
    acc_rotated = random_rotation.apply(acc_data)
    
    quat_scipy_format = quat_data[:, [1,2,3,0]]  # [rot_x, rot_y, rot_z, rot_w]
    original_rotations = R.from_quat(quat_scipy_format)
    
    rotated_rotations = random_rotation * original_rotations
    
    quat_rotated_scipy = rotated_rotations.as_quat()
    
    quat_rotated = np.column_stack([
        quat_rotated_scipy[:, 3],  # w
        quat_rotated_scipy[:, 0],  # x  
        quat_rotated_scipy[:, 1],  # y
        quat_rotated_scipy[:, 2]   # z
    ])
    
    augmented_data = np.concatenate([acc_rotated, quat_rotated], axis=1)
    
    return augmented_data

def apply_tta(batch, tta_strategies):
    augmented_batches = [batch]  # original batch

    for strategy_name, strategy_params in tta_strategies.items():
        aug_batch = {}

        for key, tensor in batch.items():
            if key in ['target', 'aux_target', 'demographics', 'main_target', 
                      'orientation_aux_target', 'seq_type_aux_target', 
                      'behavior_aux_target', 'phase_aux_target', 'pad_mask',
                      'time_pos', 'demography_bin', 'demography_cont',
                      'imu_diff', 'thm_diff', 'tof_diff']:
                aug_batch[key] = tensor
                continue

            aug_data = tensor.clone()
            
            if key not in strategy_params.get('sensors', ['imu', 'thm', 'tof']):
                aug_batch[key] = aug_data
                continue

            acc_only_augs = ['jitter', 'magnitude_warp', 'scaling']
            is_imu_acc_only_aug = (key == 'imu' and strategy_name in acc_only_augs)

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
            elif strategy_name == 'rotation':
                max_angle = strategy_params.get('max_angle', 15)
                aug_func = lambda x: apply_rotation_augmentation_numpy(x, max_angle=max_angle)
            else:
                aug_batch[key] = aug_data
                continue

            if len(aug_data.shape) == 4:
                # aug_data: (batch_size, n_sensors, seq_len, n_vars)
                batch_size, n_sensors, seq_len, n_vars = aug_data.shape
                
                for batch_idx in range(batch_size):
                    for sensor_idx in range(n_sensors):
                        sensor_data = aug_data[batch_idx, sensor_idx].numpy()
                        
                        if is_imu_acc_only_aug:
                            if sensor_data.shape[1] >= 7:
                                acc_part = sensor_data[:, :3]
                                quat_part = sensor_data[:, 3:]
                                
                                acc_augmented = aug_func(acc_part)
                                augmented_sensor_data = np.concatenate([acc_augmented, quat_part], axis=1)
                            else:
                                augmented_sensor_data = aug_func(sensor_data)
                        else:
                            if strategy_name == 'rotation' and key != 'imu':
                                augmented_sensor_data = sensor_data
                            else:
                                augmented_sensor_data = aug_func(sensor_data)
                        
                        aug_data[batch_idx, sensor_idx] = torch.tensor(
                            augmented_sensor_data, dtype=torch.float32
                        )
            
            elif len(aug_data.shape) == 3:
                # aug_data: (batch_size, seq_len, n_vars)
                batch_size, seq_len, n_vars = aug_data.shape
                
                for batch_idx in range(batch_size):
                    sensor_data = aug_data[batch_idx].numpy()
                    
                    if strategy_name == 'rotation' and key != 'imu':
                        continue
                    
                    if is_imu_acc_only_aug:
                        if sensor_data.shape[1] >= 7:
                            acc_part = sensor_data[:, :3]
                            quat_part = sensor_data[:, 3:]
                            
                            acc_augmented = aug_func(acc_part)
                            augmented_sensor_data = np.concatenate([acc_augmented, quat_part], axis=1)
                        else:
                            augmented_sensor_data = aug_func(sensor_data)
                    else:
                        augmented_sensor_data = aug_func(sensor_data)
                    
                    aug_data[batch_idx] = torch.tensor(
                        augmented_sensor_data, dtype=torch.float32
                    )
            
            aug_batch[key] = aug_data

        augmented_batches.append(aug_batch)

    return augmented_batches
