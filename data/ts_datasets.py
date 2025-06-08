import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from configs.config import cfg
from data.ts_augmentations import jitter, magnitude_warp, time_warp, scaling

class TS_CMIDataset(Dataset):
    def __init__(self, dataframe, seq_len=100, target_col='gesture', train=True):
        self.df = dataframe.reset_index(drop=True)
        self.seq_len = seq_len
        self.target_col = target_col
        self.train = train
        
        self.imu_cols = cfg.imu_cols
        self.thm_cols = cfg.thm_cols
        self.tof_cols = cfg.tof_cols
    
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

    def _normalize_sensor_data(self, data):
        if not cfg.norm_ts:
            return data
            
        normalized_data = data.copy()
        for i in range(data.shape[1]):
            column_data = data[:, i]
            if np.all(column_data == 0) or np.all(np.isnan(column_data)):
                continue
            
            mean_val = np.nanmean(column_data)
            std_val = np.nanstd(column_data)
            
            if std_val > 1e-8:
                normalized_data[:, i] = (column_data - mean_val) / std_val
        
        return normalized_data

    def _prepare_sensor_data(self, row, sensor_cols, sensor_type):
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
                s_filled = s.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill').fillna(0.0)
                data_stacked[:, i] = s_filled.values
        
        data_stacked = self._apply_augmentations(data_stacked, sensor_type)
        data_stacked = self._normalize_sensor_data(data_stacked)

        return data_stacked # shape: (seq_len, len(sensor_cols))
    
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
        
        if self.target_col and self.target_col in self.df.columns and self.target_col in row:
            features['target'] = torch.tensor(row[self.target_col], dtype=torch.long)
        
        return features
    
class TS_CMIDataset_DecomposeWHAR(TS_CMIDataset):
    def __init__(self, dataframe, seq_len=100, target_col='gesture', train=True):
        super().__init__(dataframe, seq_len, target_col, train)
    
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
        
        if 'target' in features:
            result['target'] = features['target']
            
        return result
    
class TS_CMIDataset_DecomposeWHAR_Megasensor(TS_CMIDataset):
    def __init__(self, dataframe, seq_len=100, target_col='gesture', train=True):
        super().__init__(dataframe, seq_len, target_col, train)
    
    def __getitem__(self, idx):
        features = super().__getitem__(idx)
        
        all_sensors = torch.cat([
            features['imu'], # (seq_len, 7)
            features['thm'], # (seq_len, 5) 
            features['tof']  # (seq_len, 320)
        ], dim=1) # (seq_len, 332)
        
        model_input = all_sensors.unsqueeze(0) # (1, seq_len, 332)
         
        result = {'megasensor': model_input}
        if 'target' in features:
            result['target'] = features['target']
            
        return result
    
class TS_Demo_CMIDataset(Dataset):
    def __init__(self, dataframe, seq_len=100, target_col='gesture', train=True):
        self.df = dataframe.reset_index(drop=True)
        self.seq_len = seq_len
        self.target_col = target_col
        self.train = train
       
        self.imu_cols = cfg.imu_cols
        self.thm_cols = cfg.thm_cols
        self.tof_cols = cfg.tof_cols
        self.demo_cols = cfg.demo_cols
       
        self._normalize_demographics()
   
    def _normalize_demographics(self):
        continuous_cols = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
        
        for col in continuous_cols:
            if col in self.df.columns and col in self.demo_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                self.df[col] = (self.df[col] - mean_val) / (std_val + 1e-8)

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

    def _normalize_sensor_data(self, data):
        if not cfg.norm_ts:
            return data
            
        normalized_data = data.copy()
        for i in range(data.shape[1]):
            column_data = data[:, i]
            if np.all(column_data == 0) or np.all(np.isnan(column_data)):
                continue
            
            mean_val = np.nanmean(column_data)
            std_val = np.nanstd(column_data)
            
            if std_val > 1e-8:
                normalized_data[:, i] = (column_data - mean_val) / std_val
        
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
   
    def _prepare_sensor_data(self, row, sensor_cols, sensor_type):
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
                s_filled = s.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill').fillna(0.0)
                data_stacked[:, i] = s_filled.values
        
        data_stacked = self._apply_augmentations(data_stacked, sensor_type)
        data_stacked = self._normalize_sensor_data(data_stacked)

        return data_stacked # shape: (seq_len, len(sensor_cols))
   
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
       
        if self.target_col and self.target_col in row.index:
            features['target'] = torch.tensor(row[self.target_col], dtype=torch.long)
       
        return features