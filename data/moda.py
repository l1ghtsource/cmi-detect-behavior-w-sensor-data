import numpy as np
from scipy.spatial.transform import Rotation as R

# idea from https://openaccess.thecvf.com/content/CVPR2025/supplemental/Wu_MODA_Motion-Drift_Augmentation_CVPR_2025_supplemental.pdf

def moda_augmentation(data, drift_params=None):
    if drift_params is None:
        drift_params = _generate_moda_params(data.shape[0])
    
    acc_data = data[:, :3]  # (seq_len, 3)
    quat_data = data[:, 3:]  # (seq_len, 4): [w, x, y, z]
    
    acc_augmented = _apply_moda_to_acceleration(acc_data, drift_params)
    quat_augmented = _apply_moda_to_rotation(quat_data, drift_params)
    
    augmented_data = np.concatenate([acc_augmented, quat_augmented], axis=1)
    
    return augmented_data

def _generate_moda_params(seq_len):
    mu_rot_options = [np.pi/64, -np.pi/64, np.pi/64, -np.pi/64]
    sigma_rot_options = [np.pi/32, np.pi/32, -np.pi/32, -np.pi/32]
    
    mu_acc_options = [0.5, -0.5, 0.1, -0.1]
    sigma_acc_options = [1.0, 1.0, 1.0, 1.0]
    
    variant = np.random.randint(0, 4)
    
    mu_rot = mu_rot_options[variant]
    sigma_rot = abs(sigma_rot_options[variant])
    mu_acc = mu_acc_options[variant]
    sigma_acc = sigma_acc_options[variant]
    
    delta_theta = np.random.normal(mu_rot, sigma_rot, (seq_len, 3))
    
    delta_d = np.random.normal(mu_acc, sigma_acc, (seq_len, 3))
    
    delta_theta = _smooth_drift(delta_theta)
    delta_d = _smooth_drift(delta_d)
    
    return {
        'rotation_drift': delta_theta,  # (seq_len, 3)
        'translation_drift': delta_d   # (seq_len, 3)
    }

def _smooth_drift(drift_sequence, alpha=0.7):
    smoothed = np.zeros_like(drift_sequence)
    smoothed[0] = drift_sequence[0]
    
    for t in range(1, len(drift_sequence)):
        smoothed[t] = alpha * smoothed[t-1] + (1 - alpha) * drift_sequence[t]
    
    return smoothed

def _apply_moda_to_acceleration(acc_data, drift_params):
    translation_drift = drift_params['translation_drift']
    rotation_drift = drift_params['rotation_drift']
    
    augmented_acc = np.zeros_like(acc_data)
    
    for t in range(len(acc_data)):
        acc_original = acc_data[t]
        
        rotation_angles = rotation_drift[t] * 0.1
        rotation_matrix = R.from_euler('xyz', rotation_angles).as_matrix()
        
        acc_rotated = rotation_matrix @ acc_original
        
        sliding_effect = translation_drift[t] * 0.05
        
        augmented_acc[t] = acc_rotated + sliding_effect
    
    return augmented_acc

def _apply_moda_to_rotation(quat_data, drift_params):
    rotation_drift = drift_params['rotation_drift']
    
    quat_scipy_format = quat_data[:, [1,2,3,0]]  # [rot_x, rot_y, rot_z, rot_w]
    original_rotations = R.from_quat(quat_scipy_format)
    
    augmented_rotations = []
    
    for t in range(len(quat_data)):
        original_rot = original_rotations[t]
        
        drift_angles = rotation_drift[t]
        drift_rotation = R.from_euler('xyz', drift_angles)
        
        augmented_rotation = drift_rotation * original_rot
        
        augmented_rotations.append(augmented_rotation)
    
    augmented_rotations_obj = R.from_quat([r.as_quat() for r in augmented_rotations])
    quat_augmented_scipy = augmented_rotations_obj.as_quat()
    
    quat_augmented = np.column_stack([
        quat_augmented_scipy[:, 3],  # w
        quat_augmented_scipy[:, 0],  # x  
        quat_augmented_scipy[:, 1],  # y
        quat_augmented_scipy[:, 2]   # z
    ])
    
    return quat_augmented
