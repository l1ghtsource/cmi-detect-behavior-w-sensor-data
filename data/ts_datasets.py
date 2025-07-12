import random
import numpy as np
import pandas as pd
import torch
import pywt
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from data.ts_augmentations import (
    jitter, 
    magnitude_warp,
    time_warp, 
    scaling,
    window_slice,
    window_warp,
    permutation,
    rotation,
    time_mask,
    feature_mask
)
from data.moda import moda_augmentation
from utils.denoising import apply_denoising
from utils.data_preproc import (
    remove_gravity_from_acc_df, 
    calculate_angular_velocity_from_quat, 
    calculate_angular_distance
)
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
        main_clpsd_target=cfg.main_clpsd_target,
        phase_aux_target=cfg.phase_aux_target,
        train=True, 
        norm_stats=None
    ):
        self.df = dataframe.copy().reset_index(drop=True)
        self.seq_len = seq_len
        self.main_target = main_target
        self.orientation_aux_target = orientation_aux_target
        self.main_clpsd_target = main_clpsd_target
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
        phase_array = np.asarray(phase_sequence, dtype=np.float64)
        phase_processed, _ = self._pad_or_truncate(phase_array, self.seq_len)
        gesture_indices = np.where(phase_processed == 2)[0]
        gesture_start = gesture_indices[0] / self.seq_len if len(gesture_indices) > 0 else 0.0 # FIXME: recalc it when use time_warp
        return gesture_start

    def _compute_behaviour_seq(self, behaviour_sequence):
        behaviour_array = np.asarray(behaviour_sequence, dtype=np.float64)
        behaviour_processed, _ = self._pad_or_truncate(behaviour_array, self.seq_len)
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
        original_lengths = []
        
        for col_name in sensor_cols:
            series = row[col_name]
            series_array = np.asarray(series, dtype=np.float64)
            original_lengths.append(len(series_array))
            processed_series_list.append(series_array)
        
        assert len(set(original_lengths)) == 1, f"inconsistent lengths in {sensor_type}: {original_lengths}"
        
        data_stacked = np.stack(processed_series_list, axis=1)
        
        for i in range(data_stacked.shape[1]):
            column_data = data_stacked[:, i]
            if np.all(np.isnan(column_data)):
                data_stacked[:, i] = 0.0
            elif np.any(np.isnan(column_data)):
                s = pd.Series(column_data)
                s_filled = s.interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0.0)
                data_stacked[:, i] = s_filled.values
        
        return data_stacked

    def _normalize_sensor_data(self, data, sensor_type):
        if self.norm_stats is None or sensor_type not in self.norm_stats:
            return data
            
        stats = self.norm_stats[sensor_type]
        normalized_data = (data - stats['mean']) / stats['std']
        
        return normalized_data

    def __len__(self):
        return len(self.df)
    
    def _pad_or_truncate_final(self, data, target_len):
        current_len = data.shape[0]
        
        if current_len > target_len:
            # left truncation (keep the last target_len elements)
            truncated_data = data[-target_len:]
            mask = np.ones(target_len, dtype=bool)
            return truncated_data, mask
        elif current_len < target_len:
            # left padding
            padding_rows = np.zeros((target_len - current_len, data.shape[1]), dtype=data.dtype)
            padded_data = np.concatenate([padding_rows, data], axis=0)
            mask = np.concatenate([
                np.zeros(target_len - current_len, dtype=bool), 
                np.ones(current_len, dtype=bool)
            ])
            return padded_data, mask
        
        mask = np.ones(target_len, dtype=bool)
        return data, mask
    
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
        
        n_applied = 0
        
        if random.random() < cfg.time_warp_proba and sensor_type in cfg.time_warp_sensors:
            data = time_warp(data, sigma=0.1, knot=3)
            n_applied += 1

        if random.random() < cfg.window_slice_proba and sensor_type in cfg.window_slice_sensors:
            data = window_slice(data, reduce_ratio=0.9)
            n_applied += 1

        if random.random() < cfg.permutation_proba and sensor_type in cfg.permutation_sensors:
            data = permutation(data, max_segments=5, seg_mode="equal")
            n_applied += 1

        if random.random() < cfg.time_mask_proba and sensor_type in cfg.time_mask_sensors:
            data = time_mask(data, 
                            n_features=cfg.time_mask_n_features,
                            max_width=int(cfg.time_mask_max_width_frac * self.seq_len))
            n_applied += 1
            
        if random.random() < cfg.feature_mask_proba and sensor_type in cfg.feature_mask_sensors:
            data = feature_mask(data, n_features=cfg.feature_mask_n_features)
            n_applied += 1
        
        if sensor_type == 'imu':
            return self._apply_imu_augmentations(data, n_applied)
        else:
            augmentations = []
            if random.random() < cfg.jitter_proba and sensor_type in cfg.jitter_sensors:
                augmentations.append(('jitter', lambda x: jitter(x, sigma=0.05)))
            if random.random() < cfg.magnitude_warp_proba and sensor_type in cfg.magnitude_warp_sensors:
                augmentations.append(('magnitude_warp', lambda x: magnitude_warp(x, sigma=0.15, knot=3)))
            if random.random() < cfg.scaling_proba and sensor_type in cfg.scaling_sensors:
                augmentations.append(('scaling', lambda x: scaling(x, sigma=0.08)))
            if random.random() < cfg.window_warp_proba and sensor_type in cfg.window_warp_sensors:
                augmentations.append(('window_warp', lambda x: window_warp(x, window_ratio=0.1, scales=[0.5, 2.])))
            
            selected_augmentations = random.sample(augmentations, 
                                                min(len(augmentations), cfg.max_augmentations_per_sample - n_applied))
            
            for _, aug_func in selected_augmentations:
                data = aug_func(data)
                
            return data

    def _apply_imu_augmentations(self, data, n_applied):
        available_augmentations = []

        if random.random() < cfg.moda_proba and 'imu' in cfg.moda_sensors:
            available_augmentations.append('moda')
        
        if random.random() < cfg.rotation_proba and 'imu' in cfg.rotation_sensors:
            available_augmentations.append('rotation')
        
        acc_only_augs = []
        if random.random() < cfg.jitter_proba and 'imu' in cfg.jitter_sensors:
            acc_only_augs.append('jitter')
        if random.random() < cfg.magnitude_warp_proba and 'imu' in cfg.magnitude_warp_sensors:
            acc_only_augs.append('magnitude_warp')
        if random.random() < cfg.scaling_proba and 'imu' in cfg.scaling_sensors:
            acc_only_augs.append('scaling')
        if random.random() < cfg.window_warp_proba and 'imu' in cfg.window_warp_sensors:
            acc_only_augs.append('window_warp')
        
        max_acc_augs = cfg.max_augmentations_per_sample - len(available_augmentations) - n_applied
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
            elif aug == 'jitter':
                augmented_data[:, :3] = jitter(augmented_data[:, :3], sigma=0.05)
            elif aug == 'magnitude_warp':
                augmented_data[:, :3] = magnitude_warp(augmented_data[:, :3], sigma=0.15, knot=3)
            elif aug == 'scaling':
                augmented_data[:, :3] = scaling(augmented_data[:, :3], sigma=0.08)
            elif aug == 'window_warp':
                augmented_data[:, :3] = window_warp(augmented_data[:, :3], window_ratio=0.1, scales=[0.5, 2.])
        
        return augmented_data
    
    def _apply_rotation_augmentation(self, data):
        acc_data = data[:, :3]  # (seq_len, 3)
        quat_data = data[:, 3:7]  # (seq_len, 4): [w, x, y, z]
        remaining_data = data[:, 7:] if data.shape[1] > 7 else None

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

        if remaining_data is not None:
            augmented_data = np.concatenate([acc_rotated, quat_rotated, remaining_data], axis=1)
        else:
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
    
    def _denoise_sensor_data(self, data):
        if cfg.denoise_data == 'none':
            return data
        
        denoised_data = data.copy()
        
        for feature_idx in range(data.shape[1]):
            if data.shape[0] > 1:
                denoised_valid = apply_denoising(
                    data[:, feature_idx], 
                    method=cfg.denoise_data
                )
                denoised_data[:, feature_idx] = denoised_valid
        
        return denoised_data
    
    def _generate_features(self, data, sequence_length):
        acc_x, acc_y, acc_z = data[:, 0], data[:, 1], data[:, 2]
        rot_x, rot_y, rot_z, rot_w = data[:, 3], data[:, 4], data[:, 5], data[:, 6]
        
        additional_features = []

        if cfg.kaggle_fe:
            acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
            rot_angle = 2 * np.arccos(rot_w.clip(-1, 1))

            acc_mag_jerk = np.zeros_like(acc_mag)
            acc_mag_jerk[1:] = acc_mag[1:] - acc_mag[:-1]
            rot_angle_vel = np.zeros_like(rot_angle)
            rot_angle_vel[1:] = rot_angle[1:] - rot_angle[:-1]

            linear_accs = remove_gravity_from_acc_df(data[:, :3], data[:, 3:7])
            linear_acc_x, linear_acc_y, linear_acc_z = linear_accs[:, 0], linear_accs[:, 1], linear_accs[:, 2]

            linear_acc_mag = np.sqrt(linear_acc_x ** 2 + linear_acc_y ** 2 + linear_acc_z ** 2)
            linear_acc_mag_jerk = np.zeros_like(linear_acc_mag)
            linear_acc_mag_jerk[1:] = linear_acc_mag[1:] - linear_acc_mag[:-1]   

            angular_vels = calculate_angular_velocity_from_quat(data[:, 3:7])
            angular_vel_x, angular_vel_y, angular_vel_z = angular_vels[:, 0], angular_vels[:, 1], angular_vels[:, 2]

            angular_distance = calculate_angular_distance(data[:, 3:7])

            additional_features.extend([
                acc_mag, rot_angle,
                acc_mag_jerk, rot_angle_vel,
                linear_acc_x, linear_acc_y, linear_acc_z,
                linear_acc_mag, linear_acc_mag_jerk,
                angular_vel_x, angular_vel_y, angular_vel_z,
                angular_distance
            ])
        
        if cfg.fe_mag_ang:
            acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
            rot_mag = np.sqrt(rot_x ** 2 + rot_y ** 2 + rot_z ** 2)
            rot_angle = 2 * np.arccos(np.clip(rot_w, -1, 1))
            tilt = np.arccos(acc_z / acc_mag)
            
            additional_features.extend([acc_mag, rot_mag, rot_angle, tilt])
        
        if cfg.fe_col_diff:
            XY_acc = acc_x - acc_y
            XZ_acc = acc_x - acc_z
            YZ_acc = acc_y - acc_z
            
            additional_features.extend([XY_acc, XZ_acc, YZ_acc])

        if cfg.fe_col_prod:
            prods = []
            for acc_col in [acc_x, acc_y, acc_z]:
                for rot_col in [rot_x, rot_y, rot_z]:
                    prods.append(acc_col * rot_col)
            
            additional_features.extend(prods)
        
        if cfg.lag_lead_cum:
            for i, acc_col in enumerate([acc_x, acc_y, acc_z]):
                lag_diff = np.zeros_like(acc_col)
                lag_diff[1:] = acc_col[1:] - acc_col[:-1]
                
                lead_diff = np.zeros_like(acc_col)
                lead_diff[:-1] = acc_col[:-1] - acc_col[1:]
                
                cumsum = np.cumsum(acc_col)
                cumsum_mean = np.mean(cumsum)
                cumsum_std = np.std(cumsum) + 1e-6
                cumsum_norm = (cumsum - cumsum_mean) / cumsum_std
                
                additional_features.extend([lag_diff, lead_diff, cumsum_norm])

        if cfg.fe_angles:
            acc_angle_xy = np.arctan2(acc_y, acc_x)
            acc_angle_xz = np.arctan2(acc_z, acc_x)
            acc_angle_yz = np.arctan2(acc_z, acc_y)
            acc_direction_change = np.abs(np.diff(acc_angle_xy, prepend=acc_angle_xy[0]))

            additional_features.extend([acc_angle_xy, acc_angle_xz, acc_angle_yz, acc_direction_change])

        # take it from https://oduerr.github.io/gesture/ypr_calculations.html
        if cfg.fe_euler:
            roll = np.arctan2(2 * (rot_w * rot_x + rot_y * rot_z), 1 - 2 * (rot_x ** 2 + rot_y ** 2))
            pitch = np.arcsin(np.clip(2 * (rot_w * rot_y - rot_z * rot_x), -1, 1))
            yaw = np.arctan2(2 * (rot_w * rot_z + rot_x * rot_y), 1 - 2 * (rot_y ** 2 + rot_z ** 2))

            additional_features.extend([roll, pitch, yaw])

        if cfg.fe_freq_wavelet:
            freq_wavelet_features = []
            
            for signal in [acc_x, acc_y, acc_z]:
                fft_coeffs = np.fft.rfft(signal)
                fft_magnitude = np.abs(fft_coeffs)
                freq_wavelet_features.extend(fft_magnitude[:3])
            
            for signal in [acc_x, acc_y, acc_z]:
                coeffs = pywt.wavedec(signal, 'db1', level=2)
                cA2, cD2, cD1 = coeffs
                freq_wavelet_features.extend([
                    np.mean(cA2), np.std(cA2),
                    np.mean(cD2), np.std(cD2)
                ])
            
            freq_wavelet_array = np.array(freq_wavelet_features)
            freq_wavelet_repeated = np.tile(freq_wavelet_array, (sequence_length, 1))
            
            additional_features.extend([freq_wavelet_repeated[:, i] for i in range(freq_wavelet_repeated.shape[1])])

        if cfg.fe_gravity:
            gravity_x = 2 * (rot_x * rot_z - rot_w * rot_y)
            gravity_y = 2 * (rot_w * rot_x + rot_y * rot_z)  
            gravity_z = rot_w ** 2 - rot_x ** 2 - rot_y ** 2 + rot_z ** 2
            
            acc_vertical = acc_x * gravity_x + acc_y * gravity_y + acc_z * gravity_z
            acc_horizontal_x = acc_x - acc_vertical * gravity_x
            acc_horizontal_y = acc_y - acc_vertical * gravity_y
            acc_horizontal_z = acc_z - acc_vertical * gravity_z
            acc_horizontal_mag = np.sqrt(acc_horizontal_x ** 2 + acc_horizontal_y ** 2 + acc_horizontal_z ** 2)
            
            additional_features.extend([
                gravity_x, gravity_y, gravity_z,
                acc_vertical,
                acc_horizontal_mag
            ])

        if additional_features:
            additional_features = np.column_stack(additional_features)  # (seq_len, n_new_features)
            enhanced_data = np.concatenate([data, additional_features], axis=1)
        else:
            enhanced_data = data
        
        return enhanced_data
    
    def _prepare_sensor_data(self, row, sensor_cols, sensor_type):
        data_stacked = self._prepare_sensor_data_raw(row, sensor_cols, sensor_type)
        
        data_stacked = self._denoise_sensor_data(data_stacked)
        
        if self.train:
            data_stacked = self._apply_augmentations(data_stacked, sensor_type)
        
        # if sensor_type == 'imu':
        #     data_stacked = self._generate_features(data_stacked, data_stacked.shape[0])
        
        data_stacked, padding_mask = self._pad_or_truncate_final(data_stacked, self.seq_len)
        
        data_stacked = self._normalize_sensor_data(data_stacked, sensor_type)
        
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
            features['main_clpsd_target'] = torch.tensor(row[self.main_clpsd_target], dtype=torch.long)
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
        main_clpsd_target=cfg.main_clpsd_target,
        phase_aux_target=cfg.phase_aux_target,
        train=True, 
        norm_stats=None
    ):
        super().__init__(dataframe, seq_len, main_target, orientation_aux_target, seq_type_aux_target, behavior_aux_target, main_clpsd_target, phase_aux_target, train, norm_stats)
    
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

        if cfg.use_demo:
            result['demography_bin'] = features['demography_bin'] # (3,)
            result['demography_cont'] = features['demography_cont'] # (4,)

        if self.has_target:
            result['phase_aux_target'] = features['phase_aux_target']
            result['behavior_aux_target'] = features['behavior_aux_target']
        
        if self.has_target:
            result['main_target'] = features['main_target']
            result['main_clpsd_target'] = features['main_clpsd_target']
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
        main_clpsd_target=cfg.main_clpsd_target,
        phase_aux_target=cfg.phase_aux_target,
        train=True, 
        norm_stats=None
    ):
        super().__init__(dataframe, seq_len, main_target, orientation_aux_target, seq_type_aux_target, behavior_aux_target, main_clpsd_target, phase_aux_target, train, norm_stats)

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

        if cfg.use_demo:
            result['demography_bin'] = features['demography_bin'] # (3)
            result['demography_cont'] = features['demography_cont'] # (4)

        if self.has_target:
            result['phase_aux_target'] = features['phase_aux_target']
            result['behavior_aux_target'] = features['behavior_aux_target']
        
        if self.has_target:
            result['main_target'] = features['main_target']
            result['main_clpsd_target'] = features['main_clpsd_target']
            result['orientation_aux_target'] = features['orientation_aux_target']
            result['seq_type_aux_target'] = features['seq_type_aux_target']
            
        return result