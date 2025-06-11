import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from configs.config import cfg
from data.ts_augmentations import jitter, magnitude_warp, time_warp, scaling

# classic ts dataset
class TS_CMIDataset(Dataset):
    def __init__(self, dataframe, seq_len=cfg.seq_len, target_col=cfg.target, aux_target_col=cfg.aux_target, train=True, norm_stats=None):
        self.df = dataframe.copy().reset_index(drop=True)
        self.seq_len = seq_len
        self.aux_target_col = aux_target_col
        self.target_col = target_col
        self.train = train
        self.has_target = self.target_col in self.df.columns and self.aux_target_col in self.df.columns
        
        self.imu_cols = cfg.imu_cols
        self.thm_cols = cfg.thm_cols
        self.tof_cols = cfg.tof_cols
        
        self.has_behavior = 'behavior' in self.df.columns
        
        if cfg.norm_ts:
            if norm_stats is None and train:
                self.norm_stats = self._compute_normalization_stats()
            elif norm_stats is not None:
                self.norm_stats = norm_stats
            else:
                self.norm_stats = None
        else:
            self.norm_stats = None

    def _compute_phase_moments(self, behavior_sequence):
        behavior_array = np.array(behavior_sequence)
        
        pause_indices = np.where(behavior_array == 'Pause')[0]
        gesture_indices = np.where(behavior_array == 'Gesture')[0]
        
        pause_start = pause_indices[0] / len(behavior_array) if len(pause_indices) > 0 else -1.0
        gesture_start = gesture_indices[0] / len(behavior_array) if len(gesture_indices) > 0 else -1.0
        
        return pause_start, gesture_start

    def _compute_normalization_stats(self):
        stats = {}
        
        for sensor_type, sensor_cols in [('imu', self.imu_cols), ('thm', self.thm_cols), ('tof', self.tof_cols)]:
            all_data = []
            
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                sensor_data = self._prepare_sensor_data_raw(row, sensor_cols, sensor_type)
                all_data.append(sensor_data)
            
            combined_data = np.concatenate(all_data, axis=0)  # (total_samples, n_features)
            
            means = np.nanmean(combined_data, axis=0)
            stds = np.nanstd(combined_data, axis=0)
            stds[stds < 1e-8] = 1.0
            
            stats[sensor_type] = {'mean': means, 'std': stds}
        
        return stats

    def _prepare_sensor_data_raw(self, row, sensor_cols, sensor_type):
        processed_series_list = []
        for col_name in sensor_cols:
            series = row[col_name]
            # replace -1 with 255 for tof columns
            if col_name in self.tof_cols:
                series = np.array(series)
                series[series == -1] = 255
            padded_truncated_series = self._pad_or_truncate(series, self.seq_len)
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
        
        return data_stacked

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
            return series_data[-target_len:]
        elif current_len < target_len:
            padding = np.zeros(target_len - current_len, dtype=series_data.dtype)
            return np.concatenate((padding, series_data))  # padding on the left
        return series_data

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
        data_stacked = self._prepare_sensor_data_raw(row, sensor_cols, sensor_type)
        if self.train:
            data_stacked = self._apply_augmentations(data_stacked, sensor_type)
        data_stacked = self._normalize_sensor_data(data_stacked, sensor_type)
        return data_stacked  # shape: (seq_len, len(sensor_cols))
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        imu_data = self._prepare_sensor_data(row, self.imu_cols, 'imu')
        thm_data = self._prepare_sensor_data(row, self.thm_cols, 'thm')
        tof_data = self._prepare_sensor_data(row, self.tof_cols, 'tof')
        
        features = {
            'imu': torch.tensor(imu_data, dtype=torch.float32),
            'thm': torch.tensor(thm_data, dtype=torch.float32),
            'tof': torch.tensor(tof_data, dtype=torch.float32)
        }
        
        if self.has_behavior:
            pause_start, gesture_start = self._compute_phase_moments(row['behavior'])
            features['pause_start'] = torch.tensor(pause_start, dtype=torch.float32)
            features['gesture_start'] = torch.tensor(gesture_start, dtype=torch.float32)
        
        if self.has_target:
            features['target'] = torch.tensor(row[self.target_col], dtype=torch.long)
            features['aux_target'] = torch.tensor(row[self.aux_target_col], dtype=torch.long)
        
        return features

# compatitable w/ timemil and decomposewhar !!
class TS_CMIDataset_DecomposeWHAR(TS_CMIDataset):
    def __init__(self, dataframe, seq_len=cfg.seq_len, target_col=cfg.target, aux_target_col=cfg.aux_target, train=True, norm_stats=None):
        super().__init__(dataframe, seq_len, target_col, aux_target_col, train, norm_stats)
    
    def __getitem__(self, idx):
        features = super().__getitem__(idx)
        
        imu_data = features['imu'].unsqueeze(0) # (1, seq_len, 7)
        thm_data = features['thm'].transpose(0, 1).unsqueeze(-1) # (5, seq_len, 1)
        tof_tensor = features['tof'] # (seq_len, 320)
        tof_reshaped = tof_tensor.view(-1, 5, 64).transpose(0, 1) # (5, seq_len, 64)
        
        result = {
            'imu': imu_data,
            'thm': thm_data, 
            'tof': tof_reshaped
        }

        if 'pause_start' in features:
            result['pause_start'] = features['pause_start']
            result['gesture_start'] = features['gesture_start']
        
        if 'target' in features:
            result['target'] = features['target']
            result['aux_target'] = features['aux_target']
            
        return result

# ebaniy kal, prosto zalupa
class TS_CMIDataset_DecomposeWHAR_Megasensor(TS_CMIDataset):
    def __init__(self, dataframe, seq_len=cfg.seq_len, target_col=cfg.target, aux_target_col=cfg.aux_target, train=True, norm_stats=None):
        super().__init__(dataframe, seq_len, target_col, aux_target_col, train, norm_stats)
    
    def __getitem__(self, idx):
        features = super().__getitem__(idx)
        
        all_sensors = torch.cat([
            features['imu'], # (seq_len, 7)
            features['thm'], # (seq_len, 5) 
            features['tof']  # (seq_len, 320)
        ], dim=1) # (seq_len, 332)
        
        model_input = all_sensors.unsqueeze(0) # (1, seq_len, 332)
         
        result = {'megasensor': model_input}

        if 'pause_start' in features:
            result['pause_start'] = features['pause_start']
            result['gesture_start'] = features['gesture_start']

        if 'target' in features:
            result['target'] = features['target']
            result['aux_target'] = features['aux_target']
            
        return result

# classic ds but w/ demography
class TS_Demo_CMIDataset(Dataset):
    def __init__(self, dataframe, seq_len=cfg.seq_len, target_col=cfg.target, aux_target_col=cfg.aux_target, train=True, norm_stats=None):
        self.df = dataframe.copy().reset_index(drop=True)
        self.seq_len = seq_len
        self.target_col = target_col
        self.aux_target_col = aux_target_col
        self.train = train
        self.has_target = self.target_col in self.df.columns and self.aux_target_col in self.df.columns
       
        self.imu_cols = cfg.imu_cols
        self.thm_cols = cfg.thm_cols
        self.tof_cols = cfg.tof_cols
        self.demo_cols = cfg.demo_cols
        
        self.has_behavior = 'behavior' in self.df.columns
       
        if norm_stats is None and train:
            self.norm_stats = self._compute_all_normalization_stats()
        elif norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            self.norm_stats = None

        self._normalize_demographics()

    def _compute_phase_moments(self, behavior_sequence):
        behavior_array = np.array(behavior_sequence)
        
        pause_indices = np.where(behavior_array == 'Pause')[0]
        gesture_indices = np.where(behavior_array == 'Gesture')[0]
        
        pause_start = pause_indices[0] / len(behavior_array) if len(pause_indices) > 0 else -1.0
        gesture_start = gesture_indices[0] / len(behavior_array) if len(gesture_indices) > 0 else -1.0
        
        return pause_start, gesture_start
   
    def _compute_all_normalization_stats(self):
        stats = {}
        stats['demographics'] = self._compute_demographics_stats()
        stats['sensors'] = self._compute_sensor_normalization_stats()
        return stats
    
    def _compute_demographics_stats(self):
        continuous_cols = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
        demo_stats = {}
        
        for col in continuous_cols:
            if col in self.df.columns and col in self.demo_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val < 1e-8:
                    std_val = 1.0
                demo_stats[col] = {'mean': mean_val, 'std': std_val}
        
        return demo_stats
    
    def _compute_sensor_normalization_stats(self):
        sensor_stats = {}
        
        for sensor_type, sensor_cols in [('imu', self.imu_cols), ('thm', self.thm_cols), ('tof', self.tof_cols)]:
            all_data = []
            
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                sensor_data = self._prepare_sensor_data_raw(row, sensor_cols, sensor_type)
                all_data.append(sensor_data)
            
            combined_data = np.concatenate(all_data, axis=0)
            
            means = np.nanmean(combined_data, axis=0)
            stds = np.nanstd(combined_data, axis=0)
            stds[stds < 1e-8] = 1.0
            
            sensor_stats[sensor_type] = {'mean': means, 'std': stds}
        
        return sensor_stats

    def _normalize_demographics(self):
        if self.norm_stats is None or 'demographics' not in self.norm_stats:
            return
            
        demo_stats = self.norm_stats['demographics']
        for col, stats in demo_stats.items():
            if col in self.df.columns:
                self.df[col] = (self.df[col] - stats['mean']) / stats['std']
    
    def _prepare_sensor_data_raw(self, row, sensor_cols, sensor_type):
        processed_series_list = []
        for col_name in sensor_cols:
            series = row[col_name]
            if col_name in self.tof_cols:
                series = np.array(series)
                series[series == -1] = 255
            padded_truncated_series = self._pad_or_truncate(series, self.seq_len)
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
        
        return data_stacked

    def _normalize_sensor_data(self, data, sensor_type):
        if (self.norm_stats is None or 
            'sensors' not in self.norm_stats or 
            sensor_type not in self.norm_stats['sensors'] or 
            not cfg.norm_ts):
            return data
            
        stats = self.norm_stats['sensors'][sensor_type]
        normalized_data = (data - stats['mean']) / stats['std']
        
        return normalized_data

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
       
    def __len__(self):
        return len(self.df)
   
    def _pad_or_truncate(self, series_data, target_len):
        series_data = np.asarray(series_data, dtype=np.float64)
        current_len = len(series_data)
        if current_len > target_len:
            return series_data[-target_len:]
        elif current_len < target_len:
            padding = np.zeros(target_len - current_len, dtype=series_data.dtype)
            return np.concatenate((padding, series_data))
        return series_data
   
    def _prepare_sensor_data(self, row, sensor_cols, sensor_type):
        data_stacked = self._prepare_sensor_data_raw(row, sensor_cols, sensor_type)
        data_stacked = self._apply_augmentations(data_stacked, sensor_type)
        data_stacked = self._normalize_sensor_data(data_stacked, sensor_type)
        return data_stacked
   
    def _get_demographics(self, row):
        demo_features = []
        for col in self.demo_cols:
            if col in row.index:
                value = row[col]
                if pd.isna(value):
                    demo_features.append(0.0)
                else:
                    demo_features.append(float(value))
            else:
                demo_features.append(0.0)
       
        return np.array(demo_features, dtype=np.float32)
   
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
       
        imu_data = self._prepare_sensor_data(row, self.imu_cols, 'imu')
        thm_data = self._prepare_sensor_data(row, self.thm_cols, 'thm')
        tof_data = self._prepare_sensor_data(row, self.tof_cols, 'tof')
       
        demographics = self._get_demographics(row)
       
        features = {
            'imu': torch.tensor(imu_data, dtype=torch.float32),
            'thm': torch.tensor(thm_data, dtype=torch.float32),
            'tof': torch.tensor(tof_data, dtype=torch.float32),
            'demographics': torch.tensor(demographics, dtype=torch.float32)
        }
        
        if self.has_behavior:
            pause_start, gesture_start = self._compute_phase_moments(row['behavior'])
            features['pause_start'] = torch.tensor(pause_start, dtype=torch.float32)
            features['gesture_start'] = torch.tensor(gesture_start, dtype=torch.float32)
       
        if self.has_target:
            features['target'] = torch.tensor(row[self.target_col], dtype=torch.long)
            features['aux_target'] = torch.tensor(row[self.aux_target_col], dtype=torch.long)
       
        return features