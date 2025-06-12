import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import find_peaks
import scipy.stats as stats
from configs.config import cfg
from data.ts_augmentations import jitter, magnitude_warp, time_warp, scaling

# TODO: all augmentations to gpu!

# classic ts dataset
class TS_CMIDataset(Dataset):
    def __init__(
        self, 
        dataframe, 
        seq_len=cfg.seq_len, 
        target_col=cfg.target, 
        aux_target_col=cfg.aux_target, 
        aux2_target_col=cfg.aux2_target, 
        train=True, 
        norm_stats=None
    ):
        self.df = dataframe.copy().reset_index(drop=True)
        self.seq_len = seq_len
        self.target_col = target_col
        self.aux_target_col = aux_target_col
        self.aux2_target_col = aux2_target_col
        self.train = train
        
        self.imu_cols = cfg.imu_cols
        self.thm_cols = cfg.thm_cols
        self.tof_cols = cfg.tof_cols
        
        self.demo_bin_cols = cfg.demo_bin_cols
        self.demo_cont_cols = cfg.demo_cont_cols
        
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
        phase_array = np.array(phase_sequence)
        gesture_indices = np.where(phase_array == 'Gesture')[0]
        gesture_start = gesture_indices[0] / len(phase_array) if len(gesture_indices) > 0 else 0.0 # FIXME: recalc it when use time_warp
        return gesture_start
    
    def _compute_statistics_features(self, data, sensor_type):
        n_features = data.shape[1]
        stats_list = []
        
        for feature_idx in range(n_features):
            series = data[:, feature_idx]
            
            clean_series = series[~np.isnan(series)]
            clean_series = clean_series[clean_series != 0.0]
            
            if len(clean_series) == 0:
                feature_stats = [0.0] * 14
                stats_list.extend(feature_stats)
                continue
            
            mean_val = np.mean(clean_series)
            std_val = np.std(clean_series)
            max_val = np.max(clean_series)
            min_val = np.min(clean_series)
            q25 = np.percentile(clean_series, 25)
            q75 = np.percentile(clean_series, 75)
            
            if len(clean_series) > 1:
                diff_series = np.diff(clean_series)
                diff_mean = np.mean(diff_series)
                diff_std = np.std(diff_series)
            else:
                diff_mean = 0.0
                diff_std = 0.0
            
            energy = np.sum(clean_series ** 2)
            rms = np.sqrt(np.mean(clean_series ** 2))
            
            feature_stats = [
                mean_val, std_val, max_val, min_val, q25, q75,
                diff_mean, diff_std,
                energy, rms
            ]
            
            feature_stats = [x if np.isfinite(x) else 0.0 for x in feature_stats]
            stats_list.extend(feature_stats)
        
        return np.array(stats_list, dtype=np.float32)

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

    def _prepare_sensor_data(self, row, sensor_cols, sensor_type):
        data_stacked, padding_mask = self._prepare_sensor_data_raw(row, sensor_cols, sensor_type)

        if cfg.use_stats_vectors: 
            stats_features = self._compute_statistics_features(data_stacked, sensor_type) # (len(sensor_cols) * 14)
        else:
            stats_features = None

        if self.train:
            data_stacked = self._apply_augmentations(data_stacked, sensor_type)

        data_stacked = self._normalize_sensor_data(data_stacked, sensor_type) # (seq_len, len(sensor_cols))

        return data_stacked, stats_features, padding_mask 
    
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
        
        imu_data, imu_stats, pad_mask = self._prepare_sensor_data(row, self.imu_cols, 'imu')
        thm_data, thm_stats, _ = self._prepare_sensor_data(row, self.thm_cols, 'thm')
        tof_data, tof_stats, _ = self._prepare_sensor_data(row, self.tof_cols, 'tof')

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

        if cfg.use_stats_vectors:
            features['imu_stats'] = torch.tensor(imu_stats, dtype=torch.float32)
            features['thm_stats'] = torch.tensor(thm_stats, dtype=torch.float32)
            features['tof_stats'] = torch.tensor(tof_stats, dtype=torch.float32)
        
        if 'phase' in self.df.columns:
            gesture_start = self._compute_phase_moments(row['phase'])
            features['gesture_start'] = torch.tensor(gesture_start, dtype=torch.float32)
        
        if 'gesture' in self.df.columns:
            features['target'] = torch.tensor(row[self.target_col], dtype=torch.long)
            features['aux_target'] = torch.tensor(row[self.aux_target_col], dtype=torch.long)
            features['aux2_target'] = torch.tensor(row[self.aux2_target_col], dtype=torch.long)
        
        return features

# compatitable w/ timemil and decomposewhar !!
class TS_CMIDataset_DecomposeWHAR(TS_CMIDataset):
    def __init__(
        self, 
        dataframe, 
        seq_len=cfg.seq_len, 
        target_col=cfg.target, 
        aux_target_col=cfg.aux_target, 
        aux2_target_col=cfg.aux2_target, 
        train=True, 
        norm_stats=None
    ):
        super().__init__(dataframe, seq_len, target_col, aux_target_col, aux2_target_col, train, norm_stats)
    
    def __getitem__(self, idx):
        features = super().__getitem__(idx)
        
        imu_data = features['imu'].unsqueeze(0) # (1, seq_len, 7)
        thm_data = features['thm'].transpose(0, 1).unsqueeze(-1) # (5, seq_len, 1)
        tof_tensor = features['tof'] # (seq_len, 320)
        tof_reshaped = tof_tensor.view(-1, 5, 64).transpose(0, 1) # (5, seq_len, 64)
        pad_mask = features['pad_mask'] # (seq_len,)
        
        result = {
            'imu': imu_data,
            'thm': thm_data, 
            'tof': tof_reshaped,
            'pad_mask': pad_mask,
        }

        if cfg.use_demo:
            result['demography_bin'] = features['demography_bin'] # (3)
            result['demography_cont'] = features['demography_cont'] # (4)

        if cfg.use_stats_vectors:
            result['imu_stats'] = features['imu_stats'] # (10 * 7)
            result['thm_stats'] = features['thm_stats'] # (10 * 5)
            result['tof_stats'] = features['tof_stats'] # (10 * 320)

        if 'phase' in self.df.columns:
            result['gesture_start'] = features['gesture_start']
        
        if 'gesture' in self.df.columns:
            result['target'] = features['target']
            result['aux_target'] = features['aux_target']
            result['aux2_target'] = features['aux2_target']
            
        return result

# ebaniy kal, prosto zalupa
class TS_CMIDataset_DecomposeWHAR_Megasensor(TS_CMIDataset):
    def __init__(
        self, 
        dataframe, 
        seq_len=cfg.seq_len, 
        target_col=cfg.target, 
        aux_target_col=cfg.aux_target, 
        aux2_target_col=cfg.aux2_target, 
        train=True, 
        norm_stats=None
    ):
        super().__init__(dataframe, seq_len, target_col, aux_target_col, aux2_target_col, train, norm_stats)
    
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
            'pad_mask': mask_input
        }

        if cfg.use_demo:
            result['demography_bin'] = features['demography_bin'] # (3)
            result['demography_cont'] = features['demography_cont'] # (4)

        if cfg.use_stats_vectors:
            result['imu_stats'] = features['imu_stats'] # (10 * 7)
            result['thm_stats'] = features['thm_stats'] # (10 * 5)
            result['tof_stats'] = features['tof_stats'] # (10 * 320)

        if 'phase' in self.df.columns:
            result['gesture_start'] = features['gesture_start']

        if 'gesture' in self.df.columns:
            result['target'] = features['target']
            result['aux_target'] = features['aux_target']
            result['aux2_target'] = features['aux2_target']
            
        return result