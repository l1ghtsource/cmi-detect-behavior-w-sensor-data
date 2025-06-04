import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import cwt, stft, fftconvolve
from scipy.signal.wavelets import ricker
from pyts.image import GramianAngularField, MarkovTransitionField

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from configs.config import cfg
from morlet import morlet, fractional

class IM_CMIDataset(Dataset):
    def __init__(self, dataframe, seq_len=None, feature_means=None, target_col=None):
        self.df = dataframe.reset_index(drop=True)
        self.seq_len = seq_len
        self.feature_means = feature_means or {}
        self.target_col = target_col
        
        self.imu_cols = cfg.imu_cols
        self.thm_cols = cfg.thm_cols
        self.tof_cols = cfg.tof_cols
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        imu_data = self._prepare_imu_thm_data(row, self.imu_cols, 'imu')
        thm_data = self._prepare_imu_thm_data(row, self.thm_cols, 'thm')
        tof_data = self._prepare_tof_data(row)
        
        cwt_features = self._compute_all_cwt(imu_data, thm_data)
        stft_features = self._compute_all_stft(imu_data, thm_data)
        gaf_features = self._compute_all_gaf(imu_data, thm_data)
        mtf_features = self._compute_all_mtf(imu_data, thm_data)
        superlets_features = self._compute_all_superlets(imu_data, thm_data)
        tof_heatmaps = self._compute_tof_heatmaps(tof_data)
        
        features = {
            'sequence_id': row['sequence_id'],
            'imu_cwt': self._stack_and_resize(cwt_features, self.imu_cols),
            'imu_stft': self._stack_and_resize(stft_features, self.imu_cols),
            'imu_gaf': self._stack_and_resize(gaf_features, self.imu_cols),
            'imu_mtf': self._stack_and_resize(mtf_features, self.imu_cols),
            'imu_superlets': self._stack_and_resize(superlets_features, self.imu_cols),
            'thm_cwt': self._stack_and_resize(cwt_features, self.thm_cols),
            'thm_stft': self._stack_and_resize(stft_features, self.thm_cols),
            'thm_gaf': self._stack_and_resize(gaf_features, self.thm_cols),
            'thm_mtf': self._stack_and_resize(mtf_features, self.thm_cols),
            'thm_superlets': self._stack_and_resize(superlets_features, self.thm_cols),
            'tof_heatmaps': self._resize_image(self._combine_tof_heatmaps(tof_heatmaps))
        }
        
        if self.target_col and self.target_col in row:
            features['target'] = torch.tensor(row[self.target_col], dtype=torch.float32)
        
        return features
    
    def _compute_all_superlets(self, imu_data, thm_data):
        superlets_features = {}
        all_cols = self.imu_cols + self.thm_cols
        
        for i, col in enumerate(all_cols):
            x = imu_data[:, i] if i < len(self.imu_cols) else thm_data[:, i - len(self.imu_cols)]
            superlets_features[col] = torch.tensor(self.compute_superlets(x), dtype=torch.float32)
        
        return superlets_features
    
    def compute_superlets(self, x, fs=50, c1=3, ord=(3, 7), foi=None):
        if foi is None:
            foi = np.logspace(0, np.log10(fs/4), 20)
        
        inputSize = len(x)
        frequencies = foi
        orders = np.linspace(start=ord[0], stop=ord[1], num=len(foi))
        
        superlets = []
        for iFreq in range(len(frequencies)):
            centerFreq = frequencies[iFreq]
            nWavelets = int(np.ceil(orders[iFreq]))
            
            superlets.append([])
            for iWave in range(nWavelets):
                wavelet = morlet(centerFreq, (iWave + 1) * c1, fs)
                superlets[iFreq].append(wavelet)
        
        result = np.zeros((len(frequencies), inputSize), dtype=np.float64)
        
        for iFreq in range(len(frequencies)):
            poolBuffer = np.ones(inputSize, dtype=np.float64)
            
            if len(superlets[iFreq]) > 1:
                nWavelets = int(np.floor(orders[iFreq]))
                rfactor = 1.0 / nWavelets
                
                for iWave in range(nWavelets):
                    convBuffer = fftconvolve(x, superlets[iFreq][iWave], "same")
                    poolBuffer *= 2 * np.abs(convBuffer) ** 2
                
                if fractional(orders[iFreq]) != 0 and len(superlets[iFreq]) == nWavelets + 1:
                    exponent = orders[iFreq] - nWavelets
                    rfactor = 1 / (nWavelets + exponent)
                    
                    convBuffer = fftconvolve(x, superlets[iFreq][nWavelets], "same")
                    poolBuffer *= (2 * np.abs(convBuffer) ** 2) ** exponent
                
                result[iFreq, :] = poolBuffer ** rfactor
            else:
                convResult = fftconvolve(x, superlets[iFreq][0], "same")
                result[iFreq, :] = (2 * np.abs(convResult) ** 2).astype(np.float64)
        
        return result
    
    def _stack_and_resize(self, features_dict, cols):
        feature_images = []
        for col in cols:
            feature_images.append(features_dict[col])
        
        combined = torch.cat(feature_images, dim=0)
        return self._resize_image(combined)

    def _resize_image(self, image):
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0) 
            resized = F.interpolate(image, size=224, mode='bilinear', align_corners=False)
            return resized.squeeze(0).squeeze(0)
        return image
    
    def _combine_tof_heatmaps(self, tof_heatmaps):
        seq_len = tof_heatmaps.shape[1] # (5, seq_len, 8, 8) -> seq_len
        
        combined_timesteps = []
        for t in range(seq_len):
            sensors_at_t = tof_heatmaps[:, t, :, :]  # (5, 8, 8)
            
            sensor_columns = []
            for sensor_idx in range(5):
                sensor_flat = sensors_at_t[sensor_idx].flatten()  # (64,)
                sensor_columns.append(sensor_flat.unsqueeze(1))  # (64, 1)
            
            timestep_image = torch.cat(sensor_columns, dim=1)  # (64, 5)
            combined_timesteps.append(timestep_image)
        
        final_image = torch.cat(combined_timesteps, dim=1)  # (64, seq_len * 5)
        return final_image.T
    
    def _prepare_imu_thm_data(self, row, cols, data_type):
        data = np.stack([self.truncate(row[col], self.seq_len) for col in cols], axis=1)
        
        for i, col in enumerate(cols):
            x = data[:, i]
            if np.all(np.isnan(x)):
                data[:, i] = np.full_like(x, self.feature_means.get(col, 0))
            else:
                data[:, i] = pd.Series(x).interpolate().fillna(method='bfill').fillna(method='ffill').values
        
        return data
    
    def _prepare_tof_data(self, row):
        return np.stack([self.truncate(row[col], self.seq_len) for col in self.tof_cols], axis=1)
    
    def _compute_all_cwt(self, imu_data, thm_data):
        cwt_features = {}
        all_cols = self.imu_cols + self.thm_cols
        
        for i, col in enumerate(all_cols):
            x = imu_data[:, i] if i < len(self.imu_cols) else thm_data[:, i - len(self.imu_cols)]
            cwt_features[col] = torch.tensor(self.compute_cwt(x), dtype=torch.float32)
        
        return cwt_features
    
    def _compute_all_stft(self, imu_data, thm_data):
        stft_features = {}
        all_cols = self.imu_cols + self.thm_cols
        
        for i, col in enumerate(all_cols):
            x = imu_data[:, i] if i < len(self.imu_cols) else thm_data[:, i - len(self.imu_cols)]
            stft_features[col] = torch.tensor(self.compute_stft(x), dtype=torch.float32)
        
        return stft_features
    
    def _compute_all_gaf(self, imu_data, thm_data):
        gaf_features = {}
        all_cols = self.imu_cols + self.thm_cols
        
        for i, col in enumerate(all_cols):
            x = imu_data[:, i] if i < len(self.imu_cols) else thm_data[:, i - len(self.imu_cols)]
            x_norm = self.safe_normalize_gaf(x)
            gaf_features[col] = torch.tensor(self.compute_gaf(x_norm), dtype=torch.float32)
        
        return gaf_features
    
    def _compute_all_mtf(self, imu_data, thm_data):
        mtf_features = {}
        all_cols = self.imu_cols + self.thm_cols
        
        for i, col in enumerate(all_cols):
            x = imu_data[:, i] if i < len(self.imu_cols) else thm_data[:, i - len(self.imu_cols)]
            x_norm = (x - np.min(x)) / (np.ptp(x) + 1e-9)
            mtf_features[col] = torch.tensor(self.compute_mtf(x_norm, n_bins=8, image_size=len(x_norm)), dtype=torch.float32)
        
        return mtf_features
    
    def _compute_tof_heatmaps(self, tof_data):
        return torch.tensor(self.tof_to_heatmaps(tof_data), dtype=torch.float32)
    
    def truncate(self, x, seq_len):
        if seq_len is not None and len(x) > seq_len:
            return x[:seq_len]
        return x

    def safe_normalize_gaf(self, x):
        ptp = np.ptp(x)
        if ptp < 1e-9:
            return np.zeros_like(x)
        return 2 * (x - np.min(x)) / ptp - 1
    
    def compute_cwt(self, x, widths=np.arange(1, 31)):
        x = (x - x.min()) / (x.ptp() + 1e-9)
        return cwt(x, ricker, widths)
    
    def compute_stft(self, x, fs=50, nperseg=32, noverlap=16):
        x = (x - x.min()) / (x.ptp() + 1e-9)
        f, t, Zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        return np.abs(Zxx)
    
    def compute_gaf(self, x):
        gaf = GramianAngularField(method='summation')
        return gaf.fit_transform(x.reshape(1, -1))[0]
    
    def compute_mtf(self, x, n_bins=8, image_size=32):
        image_size = min(image_size, len(x))
        mtf = MarkovTransitionField(n_bins=n_bins, image_size=image_size)
        return mtf.fit_transform(x.reshape(1, -1))[0]
    
    def tof_to_heatmaps(self, tof_sequence):
        heatmaps = []
        for i in range(5):
            sensor_data = tof_sequence[:, i*64:(i+1)*64]
            sensor_data = (sensor_data - sensor_data.min()) / (sensor_data.ptp() + 1e-9)
            heatmaps_sensor = sensor_data.reshape(-1, 8, 8)
            heatmaps.append(heatmaps_sensor)
        return np.array(heatmaps)
    
def visualize_sample_images(dataset, idx=0):
    sample = dataset[idx]
    
    keys = [
        'imu_cwt', 'imu_stft', 'imu_gaf', 'imu_mtf', 'imu_superlets',
        'thm_cwt', 'thm_stft', 'thm_gaf', 'thm_mtf', 'thm_superlets',
        'tof_heatmaps'
    ]
    
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    
    for i, key in enumerate(keys[:10]):
        row = i // 5
        col = i % 5
        img = sample[key]
        
        im = axes[row, col].imshow(img, cmap='viridis', aspect='auto')
        axes[row, col].set_title(f'{key}\n{img.shape=}', fontsize=12)
        axes[row, col].axis('off')
        
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    for col in range(5):
        axes[2, col].axis('off')
    
    ax_center = plt.subplot2grid((3, 5), (2, 2), fig=fig)
    
    img = sample[keys[10]]  # 'tof_heatmaps'
    im = ax_center.imshow(img, cmap='viridis', aspect='auto')
    ax_center.set_title(f'{keys[10]}\n{img.shape=}', fontsize=12)
    ax_center.axis('off')
    
    plt.colorbar(im, ax=ax_center, fraction=0.046, pad=0.04)
    
    sequence_id = sample.get('sequence_id', 'unk')
    target = sample.get('target', 'no target')
    fig.suptitle(f'{idx=}, {sequence_id=}, {target=}', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()