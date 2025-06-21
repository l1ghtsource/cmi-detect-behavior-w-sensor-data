import random
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from data.ts_augmentations import jitter, magnitude_warp, time_warp, scaling
from data.moda import moda_augmentation
from utils.denoising import apply_denoising
from configs.config import cfg

# TODO: all augmentations to gpu!

# classic ts dataset
class TS_CMIDataset(Dataset):
    def __init__(
        self, 
        dataframe, 
        seq_len=cfg.seq_len, 
        main_target=cfg.main_target, 
        orientation_aux_target=cfg.orientation_aux_target, 
        seq_type_aux_target=cfg.seq_type_aux_target, 
        behavior_aux_target=cfg.behavior_aux_target,
        phase_aux_target=cfg.phase_aux_target,
        train=True, 
        norm_stats=None
    ):
        self.df = dataframe.copy().reset_index(drop=True)
        self.seq_len = seq_len
        self.main_target = main_target
        self.orientation_aux_target = orientation_aux_target
        self.seq_type_aux_target = seq_type_aux_target
        self.behavior_aux_target = behavior_aux_target
        self.phase_aux_target = phase_aux_target
        self.train = train
        
        self.imu_cols = cfg.imu_cols
        self.thm_cols = cfg.thm_cols
        self.tof_cols = cfg.tof_cols
        
        self.demo_bin_cols = cfg.demo_bin_cols
        self.demo_cont_cols = cfg.demo_cont_cols

        self.has_target = self.main_target in self.df.columns
        
        if cfg.norm_ts:
            if norm_stats is None and train:
                self.norm_stats = self._compute_normalization_stats()
            elif norm_stats is not None:
                self.norm_stats = norm_stats
            else:
                self.norm_stats = None
        else:
            self.norm_stats = None

    def _compute_phase_moments(self, phase_sequence):
        phase_processed, _ = self._pad_or_truncate(phase_sequence, self.seq_len)
        gesture_indices = np.where(phase_processed == 2)[0]
        gesture_start = gesture_indices[0] / self.seq_len if len(gesture_indices) > 0 else 0.0 # FIXME: recalc it when use time_warp
        return gesture_start
    
    def _compute_behaviour_seq(self, behaviour_sequence):
        behaviour_processed, _ = self._pad_or_truncate(behaviour_sequence, self.seq_len)
        return behaviour_processed

    def _compute_normalization_stats(self):
        stats = {}
        
        for sensor_type, sensor_cols in [('imu', self.imu_cols), ('thm', self.thm_cols), ('tof', self.tof_cols)]:
            all_data = []
            
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                sensor_data, _ = self._prepare_sensor_data_raw(row, sensor_cols, sensor_type)
                all_data.append(sensor_data)
            
            combined_data = np.concatenate(all_data, axis=0)  # (total_samples, n_features)
            
            mask = combined_data != 0
            means = np.array([np.nanmean(combined_data[mask[:, i], i]) if np.any(mask[:, i]) else np.nan 
                            for i in range(combined_data.shape[1])])
            stds = np.array([np.nanstd(combined_data[mask[:, i], i]) if np.any(mask[:, i]) else np.nan 
                            for i in range(combined_data.shape[1])])
            
            stds[np.isnan(stds) | (stds < 1e-8)] = 1.0
            means[np.isnan(means)] = 0.0
            
            stats[sensor_type] = {'mean': means, 'std': stds}

        if cfg.use_demo:
            demo_stats = {}
            for col in self.demo_cont_cols:
                if col in self.df.columns:
                    values = self.df[col].values
                    clean_values = values[~np.isnan(values)]
                    if len(clean_values) > 0:
                        mean_val = np.mean(clean_values)
                        std_val = np.std(clean_values)
                        std_val = std_val if std_val > 1e-8 else 1.0
                    else:
                        mean_val, std_val = 0.0, 1.0
                    demo_stats[col] = {'mean': mean_val, 'std': std_val}
            stats['demography'] = demo_stats

        return stats
    
    def _prepare_sensor_data_raw(self, row, sensor_cols, sensor_type):
        processed_series_list = []
        for col_name in sensor_cols:
            series = row[col_name]
            padded_truncated_series, mask = self._pad_or_truncate(series, self.seq_len)
            processed_series_list.append(padded_truncated_series)
        
        data_stacked = np.stack(processed_series_list, axis=1)
        
        for i in range(data_stacked.shape[1]):
            column_data = data_stacked[:, i]
            if np.all(np.isnan(column_data)):
                data_stacked[:, i] = 0.0
            elif np.any(np.isnan(column_data)):
                s = pd.Series(column_data)
                s_filled = s.interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0.0)
                data_stacked[:, i] = s_filled.values
        
        return data_stacked, mask

    def _normalize_sensor_data(self, data, sensor_type):
        if self.norm_stats is None or sensor_type not in self.norm_stats:
            return data
            
        stats = self.norm_stats[sensor_type]
        normalized_data = (data - stats['mean']) / stats['std']
        
        return normalized_data

    def __len__(self):
        return len(self.df)
    
    def _pad_or_truncate(self, series_data, target_len):
        series_data = np.asarray(series_data, dtype=np.float64)
        current_len = len(series_data)
        
        if current_len > target_len:
            # left truncation (keep the last target_len elements)
            return series_data[-target_len:], np.ones(target_len, dtype=bool)
        elif current_len < target_len:
            padding = np.zeros(target_len - current_len, dtype=series_data.dtype) # left padding
            padded_series = np.concatenate((padding, series_data))
            mask = np.concatenate((np.zeros(target_len - current_len, dtype=bool), 
                                np.ones(current_len, dtype=bool)))
            return padded_series, mask
        
        return series_data, np.ones(target_len, dtype=bool)

    def _apply_augmentations(self, data, sensor_type):
        if not self.train:
            return data
        
        if sensor_type == 'imu':
            return self._apply_imu_augmentations(data)
        else:
            augmentations = []
            if random.random() < cfg.jitter_proba and sensor_type in cfg.jitter_sensors:
                augmentations.append(('jitter', lambda x: jitter(x, sigma=0.03)))
            if random.random() < cfg.magnitude_warp_proba and sensor_type in cfg.magnitude_warp_sensors:
                augmentations.append(('magnitude_warp', lambda x: magnitude_warp(x, sigma=0.15, knot=3)))
            if random.random() < cfg.time_warp_proba and sensor_type in cfg.time_warp_sensors:
                augmentations.append(('time_warp', lambda x: time_warp(x, sigma=0.1, knot=3)))
            if random.random() < cfg.scaling_proba and sensor_type in cfg.scaling_sensors:
                augmentations.append(('scaling', lambda x: scaling(x, sigma=0.08)))
            
            selected_augmentations = random.sample(augmentations, 
                                                min(len(augmentations), cfg.max_augmentations_per_sample))
            
            for _, aug_func in selected_augmentations:
                data = aug_func(data)
                
            return data

    def _apply_imu_augmentations(self, data):
        available_augmentations = []

        if random.random() < cfg.moda_proba and 'imu' in cfg.moda_sensors:
            available_augmentations.append('moda')
        
        if random.random() < cfg.rotation_proba and 'imu' in cfg.rotation_sensors:
            available_augmentations.append('rotation')
        
        if random.random() < cfg.time_warp_proba and 'imu' in cfg.time_warp_sensors:
            available_augmentations.append('time_warp')
        
        acc_only_augs = []
        if random.random() < cfg.jitter_proba and 'imu' in cfg.jitter_sensors:
            acc_only_augs.append('jitter')
        if random.random() < cfg.magnitude_warp_proba and 'imu' in cfg.magnitude_warp_sensors:
            acc_only_augs.append('magnitude_warp')
        if random.random() < cfg.scaling_proba and 'imu' in cfg.scaling_sensors:
            acc_only_augs.append('scaling')
        
        max_acc_augs = cfg.max_augmentations_per_sample - len(available_augmentations)
        if max_acc_augs > 0 and acc_only_augs:
            selected_acc_augs = random.sample(acc_only_augs, 
                                            min(len(acc_only_augs), max_acc_augs))
            available_augmentations.extend(selected_acc_augs)
        
        augmented_data = data.copy()
        
        for aug in available_augmentations:
            if aug == 'moda':
                try:
                    augmented_data = moda_augmentation(augmented_data)
                except: # missed quat (its ok)
                    pass
            if aug == 'rotation':
                augmented_data = self._apply_rotation_augmentation(augmented_data)
            elif aug == 'time_warp':
                augmented_data = time_warp(augmented_data, sigma=0.1, knot=3)
            elif aug == 'jitter':
                augmented_data[:, :3] = jitter(augmented_data[:, :3], sigma=0.03)
            elif aug == 'magnitude_warp':
                augmented_data[:, :3] = magnitude_warp(augmented_data[:, :3], sigma=0.15, knot=3)
            elif aug == 'scaling':
                augmented_data[:, :3] = scaling(augmented_data[:, :3], sigma=0.08)
        
        return augmented_data
    
    def _apply_rotation_augmentation(self, data):
        acc_data = data[:, :3]  # (seq_len, 3)
        quat_data = data[:, 3:]  # (seq_len, 4): [w, x, y, z]

        if hasattr(cfg, 'rotation_max_angle') and cfg.rotation_max_angle:
            try:
                acc_rotated, quat_rotated = self._simple_z_rotation(acc_data, quat_data)
            except: # missed quat (its ok)
                acc_rotated, quat_rotated = acc_data, quat_data
        else:
            try:
                acc_rotated, quat_rotated = self._full_rotation_augmentation(acc_data, quat_data)
            except: # missed quat (its ok)
                acc_rotated, quat_rotated = acc_data, quat_data

        augmented_data = np.concatenate([acc_rotated, quat_rotated], axis=1)

        return augmented_data

    def _full_rotation_augmentation(self, acc_data, quat_data):
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

        return acc_rotated, quat_rotated

    def _simple_z_rotation(self, acc_data, quat_data):
        max_angle_deg = getattr(cfg, 'rotation_max_angle', 30)
        angle = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180
        
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0], 
            [0,      0,     1]
        ])
        
        acc_rotated = acc_data @ rotation_matrix.T
        
        z_rotation = R.from_euler('z', angle)
        
        quat_scipy_format = quat_data[:, [1,2,3,0]]  
        original_rotations = R.from_quat(quat_scipy_format)
        rotated_rotations = z_rotation * original_rotations
        quat_rotated_scipy = rotated_rotations.as_quat()
        
        quat_rotated = np.column_stack([
            quat_rotated_scipy[:, 3],  # w
            quat_rotated_scipy[:, 0],  # x  
            quat_rotated_scipy[:, 1],  # y
            quat_rotated_scipy[:, 2]   # z
        ])
        
        return acc_rotated, quat_rotated
    
    def _denoise_sensor_data(self, data, pad_mask):
        if cfg.denoise_data == 'none':
            return data
        
        denoised_data = data.copy()
        valid_indices = np.where(pad_mask)[0]
        
        if len(valid_indices) == 0:
            return denoised_data
        
        for feature_idx in range(data.shape[1]):
            if len(valid_indices) > 1:
                valid_data = data[valid_indices, feature_idx]
                denoised_valid = apply_denoising(
                    valid_data, 
                    method=cfg.denoise_data
                )
                denoised_data[valid_indices, feature_idx] = denoised_valid
        
        return denoised_data

    def _prepare_sensor_data(self, row, sensor_cols, sensor_type):
        data_stacked, padding_mask = self._prepare_sensor_data_raw(row, sensor_cols, sensor_type)

        data_stacked = self._denoise_sensor_data(data_stacked, padding_mask)

        if self.train:
            data_stacked = self._apply_augmentations(data_stacked, sensor_type)

        data_stacked = self._normalize_sensor_data(data_stacked, sensor_type) # (seq_len, len(sensor_cols))

        return data_stacked, padding_mask 
    
    def _prepare_demographic_data(self, row):
        demo_bin = []
        for col in self.demo_bin_cols:
            if col in row:
                val = row[col] if not pd.isna(row[col]) else 0
                demo_bin.append(float(val))
            else:
                demo_bin.append(0.0)
        
        demo_cont = []
        for col in self.demo_cont_cols:
            if col in row:
                val = row[col] if not pd.isna(row[col]) else 0.0
                
                if (self.norm_stats is not None and 
                    'demography' in self.norm_stats and 
                    col in self.norm_stats['demography']):
                    stats = self.norm_stats['demography'][col]
                    val = (val - stats['mean']) / stats['std']
                
                demo_cont.append(val)
            else:
                demo_cont.append(0.0)
        
        return np.array(demo_bin, dtype=np.float32), np.array(demo_cont, dtype=np.float32)
    
    def _generate_time_positions(self):
        positions = np.arange(1, self.seq_len + 1) / self.seq_len
        return positions
    
    def _compute_sensor_diff(self, sensor_data):
        diff_data = np.zeros_like(sensor_data)
        diff_data[1:] = sensor_data[1:] - sensor_data[:-1]
        return diff_data
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        imu_data, pad_mask = self._prepare_sensor_data(row, self.imu_cols, 'imu')
        thm_data, _ = self._prepare_sensor_data(row, self.thm_cols, 'thm')
        tof_data, _ = self._prepare_sensor_data(row, self.tof_cols, 'tof')

        features = {
            'imu': torch.tensor(imu_data, dtype=torch.float32),
            'thm': torch.tensor(thm_data, dtype=torch.float32),
            'tof': torch.tensor(tof_data, dtype=torch.float32),
            'pad_mask': torch.tensor(pad_mask, dtype=torch.float32),
        }

        if cfg.use_time_pos:
            time_pos = self._generate_time_positions()
            features['time_pos'] = torch.tensor(time_pos, dtype=torch.float32)

        if cfg.use_diff:
            imu_diff = self._compute_sensor_diff(imu_data)
            thm_diff = self._compute_sensor_diff(thm_data) 
            tof_diff = self._compute_sensor_diff(tof_data)
            features['imu_diff'] = torch.tensor(imu_diff, dtype=torch.float32)
            features['thm_diff'] = torch.tensor(thm_diff, dtype=torch.float32)
            features['tof_diff'] = torch.tensor(tof_diff, dtype=torch.float32)

        if cfg.use_demo:
            demo_bin, demo_cont = self._prepare_demographic_data(row)
            features['demography_bin'] = torch.tensor(demo_bin, dtype=torch.float32)
            features['demography_cont'] = torch.tensor(demo_cont, dtype=torch.float32)
        
        if self.has_target:
            phase_aux_target = self._compute_phase_moments(row[self.phase_aux_target])
            behavior_aux_target = self._compute_behaviour_seq(row[self.behavior_aux_target])
            features['phase_aux_target'] = torch.tensor(phase_aux_target, dtype=torch.float32)
            features['behavior_aux_target'] = torch.tensor(behavior_aux_target, dtype=torch.long)
        
        if self.has_target:
            features['main_target'] = torch.tensor(row[self.main_target], dtype=torch.long)
            features['orientation_aux_target'] = torch.tensor(row[self.orientation_aux_target], dtype=torch.long)
            features['seq_type_aux_target'] = torch.tensor(row[self.seq_type_aux_target], dtype=torch.long)
        
        return features

# compatitable w/ timemil and decomposewhar !!
class TS_CMIDataset_DecomposeWHAR(TS_CMIDataset):
    def __init__(
        self, 
        dataframe, 
        seq_len=cfg.seq_len, 
        main_target=cfg.main_target, 
        orientation_aux_target=cfg.orientation_aux_target, 
        seq_type_aux_target=cfg.seq_type_aux_target, 
        behavior_aux_target=cfg.behavior_aux_target,
        phase_aux_target=cfg.phase_aux_target,
        train=True, 
        norm_stats=None
    ):
        super().__init__(dataframe, seq_len, main_target, orientation_aux_target, seq_type_aux_target, behavior_aux_target, phase_aux_target, train, norm_stats)
    
    def __getitem__(self, idx):
        features = super().__getitem__(idx)
        
        imu_data = features['imu'].unsqueeze(0) # (1, seq_len, 7)
        thm_data = features['thm'].transpose(0, 1).unsqueeze(-1) # (5, seq_len, 1)
        tof_data = features['tof'].view(-1, 5, 64).transpose(0, 1) # (5, seq_len, 64)
        pad_mask = features['pad_mask'] # (seq_len,)
        
        result = {
            'imu': imu_data,
            'thm': thm_data, 
            'tof': tof_data,
            'pad_mask': pad_mask,
        }

        if cfg.use_time_pos:
            result['time_pos'] = features['time_pos'] # (seq_len,)

        if cfg.use_diff:
            result['imu_diff'] = features['imu_diff'].unsqueeze(0) # (1, seq_len, 7)
            result['thm_diff'] = features['thm_diff'].transpose(0, 1).unsqueeze(-1) # (5, seq_len, 1)
            result['tof_diff'] = features['tof_diff'].view(-1, 5, 64).transpose(0, 1) # (5, seq_len, 64) 

        if cfg.use_demo:
            result['demography_bin'] = features['demography_bin'] # (3,)
            result['demography_cont'] = features['demography_cont'] # (4,)

        if self.has_target:
            result['phase_aux_target'] = features['phase_aux_target']
            result['behavior_aux_target'] = features['behavior_aux_target']
        
        if self.has_target:
            result['main_target'] = features['main_target']
            result['orientation_aux_target'] = features['orientation_aux_target']
            result['seq_type_aux_target'] = features['seq_type_aux_target']
            
        return result

# ebaniy kal, prosto zalupa
class TS_CMIDataset_DecomposeWHAR_Megasensor(TS_CMIDataset):
    def __init__(
        self, 
        dataframe, 
        seq_len=cfg.seq_len, 
        main_target=cfg.main_target, 
        orientation_aux_target=cfg.orientation_aux_target, 
        seq_type_aux_target=cfg.seq_type_aux_target, 
        behavior_aux_target=cfg.behavior_aux_target,
        phase_aux_target=cfg.phase_aux_target,
        train=True, 
        norm_stats=None
    ):
        super().__init__(dataframe, seq_len, main_target, orientation_aux_target, seq_type_aux_target, behavior_aux_target, phase_aux_target, train, norm_stats)

    def __getitem__(self, idx):
        features = super().__getitem__(idx)
        
        all_sensors = torch.cat([
            features['imu'], # (seq_len, 7)
            features['thm'], # (seq_len, 5) 
            features['tof']  # (seq_len, 320)
        ], dim=1) # (seq_len, 332)
        
        model_input = all_sensors.unsqueeze(0) # (1, seq_len, 332)
        mask_input = features['pad_mask'] # (seq_len)

        result = {
            'megasensor': model_input,
            'pad_mask': mask_input,
        }

        if cfg.use_time_pos:
            result['time_pos'] = features['time_pos'] # (seq_len,)

        if cfg.use_diff:
            result['megasensor_diff'] = torch.cat([
                features['imu_diff'], # (seq_len, 7)
                features['thm_diff'], # (seq_len, 5) 
                features['tof_diff']  # (seq_len, 320)
            ], dim=1) # (seq_len, 332)

        if cfg.use_demo:
            result['demography_bin'] = features['demography_bin'] # (3)
            result['demography_cont'] = features['demography_cont'] # (4)

        if self.has_target:
            result['phase_aux_target'] = features['phase_aux_target']
            result['behavior_aux_target'] = features['behavior_aux_target']
        
        if self.has_target:
            result['main_target'] = features['main_target']
            result['orientation_aux_target'] = features['orientation_aux_target']
            result['seq_type_aux_target'] = features['seq_type_aux_target']
            
        return result