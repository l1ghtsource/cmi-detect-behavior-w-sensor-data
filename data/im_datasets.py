import random
import numpy as np
import torch
import pywt
from scipy import signal
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from configs.config import cfg
from data.ts_datasets import TS_CMIDataset

class IM_CMIDataset(TS_CMIDataset):
    def __init__(
            self, dataframe, seq_len=cfg.seq_len, main_target=cfg.main_target, seq_type_aux_target=cfg.seq_type_aux_target, 
            train=True, im_size=cfg.im_size, transform_type=cfg.transform_type
    ):
        super().__init__(dataframe, seq_len, main_target, seq_type_aux_target, train)
        self.im_size = im_size
        self.transform_type = transform_type # cwt or stft
        
    def _create_spectrogram(self, signal_data, method='cwt'):
        if method == 'cwt':
            scales = np.geomspace(1, min(64, len(signal_data)//2), num=64)
            coefs, freqs = pywt.cwt(signal_data, scales, 'cmor1.5-1.0')
            spectrogram = np.abs(coefs)
        else: # stft
            nperseg = min(32, len(signal_data)//4)
            f, t, Zxx = signal.stft(signal_data, nperseg=nperseg)
            spectrogram = np.abs(Zxx)
            
        if spectrogram.max() > 0:
            spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        
        return spectrogram
    
    def _create_imu_image(self, imu_data):
        spectrograms = []
        
        for channel in range(imu_data.shape[1]):
            channel_data = imu_data[:, channel]
            spec = self._create_spectrogram(channel_data, self.transform_type)
            spectrograms.append(spec)
        
        combined_image = np.vstack(spectrograms)
        
        return combined_image
    
    def _create_thm_image(self, thm_data):
        spectrograms = []
        
        for channel in range(thm_data.shape[1]):
            channel_data = thm_data[:, channel]
            spec = self._create_spectrogram(channel_data, self.transform_type)
            spectrograms.append(spec)
        
        combined_image = np.vstack(spectrograms)
        
        return combined_image
    
    def _create_tof_image(self, tof_data):
        # (seq_len, 320) -> (seq_len, 5, 64) -> 5 8x8 cards
        tof_reshaped = tof_data.reshape(tof_data.shape[0], 5, 64)

        if not cfg.window_tof:
            heatmaps = []
            for sensor_idx in range(5):
                sensor_data = tof_reshaped[:, sensor_idx, :] # (seq_len, 64)
                avg_data = np.mean(sensor_data, axis=0) # (64,)
                heatmap = avg_data.reshape(8, 8)
                if heatmap.max() > heatmap.min():
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                heatmaps.append(heatmap)
            combined_image = np.hstack(heatmaps) # (8, 40)
        else:
            sensor_images = []
            for sensor_idx in range(5):
                sensor_data = tof_reshaped[:, sensor_idx, :] # (seq_len, 64)
                temporal_frames = [] # temporal-spatial representation
                window_size = len(sensor_data) // 4 # 4 temporal windows
                for i in range(4):
                    start_idx = i * window_size
                    end_idx = (i + 1) * window_size if i < 3 else len(sensor_data)
                    window_data = np.mean(sensor_data[start_idx:end_idx], axis=0)
                    temporal_frames.append(window_data.reshape(8, 8))
                sensor_image = np.vstack(temporal_frames) # (32, 8)
                sensor_images.append(sensor_image)
            combined_image = np.hstack(sensor_images) # (32, 40)
 
        return combined_image
    
    def _resize_to_target(self, image):
        if image.shape[0] == 0 or image.shape[1] == 0:
            return np.zeros((self.im_size, self.im_size))
            
        zoom_factors = (self.im_size / image.shape[0], self.im_size / image.shape[1])
        
        resized_image = zoom(image, zoom_factors, order=1)
        
        if resized_image.shape != (self.im_size, self.im_size):
            resized_image = np.resize(resized_image, (self.im_size, self.im_size))
        
        return resized_image
    
    def _to_rgb_format(self, image):
        if not cfg.use_grads:
            rgb_image = np.stack([image, image, image], axis=0)
        else:
            channel1 = image # orig
            channel2 = np.gradient(image, axis=0) # time grad
            channel3 = np.gradient(image, axis=1) # freq grad
            
            channels = [channel1, channel2, channel3]
            normalized_channels = []
            
            for ch in channels:
                if ch.max() > ch.min():
                    ch_norm = (ch - ch.min()) / (ch.max() - ch.min())
                else:
                    ch_norm = np.zeros_like(ch)
                normalized_channels.append(ch_norm)
            
            rgb_image = np.stack(normalized_channels, axis=0)
        return rgb_image
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        imu_data = self._prepare_sensor_data(row, self.imu_cols, 'imu')
        thm_data = self._prepare_sensor_data(row, self.thm_cols, 'thm')
        tof_data = self._prepare_sensor_data(row, self.tof_cols, 'tof')
        
        imu_image = self._create_imu_image(imu_data)
        thm_image = self._create_thm_image(thm_data)
        tof_image = self._create_tof_image(tof_data)
        
        imu_image = self._resize_to_target(imu_image)
        thm_image = self._resize_to_target(thm_image)
        tof_image = self._resize_to_target(tof_image)
        
        imu_rgb = self._to_rgb_format(imu_image)
        thm_rgb = self._to_rgb_format(thm_image)
        tof_rgb = self._to_rgb_format(tof_image)
        
        features = {
            'im_imu': torch.tensor(imu_rgb, dtype=torch.float32),
            'im_thm': torch.tensor(thm_rgb, dtype=torch.float32),
            'im_tof': torch.tensor(tof_rgb, dtype=torch.float32)
        }
        
        if self.has_target:
            features['target'] = torch.tensor(row[self.target_col], dtype=torch.long)
            features['aux_target'] = torch.tensor(row[self.aux_target_col], dtype=torch.long)
        
        return features
    
    def visualize_samples(self, n_samples=5, figsize=(15, 10)):
        random_indices = random.sample(range(len(self)), min(n_samples, len(self)))
        
        fig, axes = plt.subplots(n_samples, 3, figsize=figsize)
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(random_indices):
            sample = self[idx]
            
            img_imu = sample['im_imu'].permute(1, 2, 0).numpy()
            img_thm = sample['im_thm'].permute(1, 2, 0).numpy()
            img_tof = sample['im_tof'].permute(1, 2, 0).numpy()
            
            target = sample.get('target', 'N/A')
            aux_target = sample.get('aux_target', 'N/A')
            
            axes[i, 0].imshow(img_imu, cmap='viridis')
            axes[i, 0].set_title(f'imu, {target=}, {aux_target=}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(img_thm, cmap='plasma')
            axes[i, 1].set_title(f'thm, {target=}, {aux_target=}')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(img_tof, cmap='hot')
            axes[i, 2].set_title(f'tof, {target=}, {aux_target=}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()