import torch
import torch.nn as nn
from typing import Optional, Callable

from modules.panns import PANNsFeatureExtractor
from configs.config import cfg

class PANNsCLF_SingleSensor_v1(nn.Module):
    def __init__(self, 
                 n_channels: int = cfg.imu_vars, 
                 seq_len: int = cfg.seq_len, 
                 num_classes: int = cfg.main_num_classes, 
                 base_filters: int | tuple = 128, 
                 kernel_sizes: tuple = (32, 16, 4, 2), 
                 stride: int = 4, 
                 sigmoid: bool = False, 
                 output_size: Optional[int] = None, 
                 conv: Callable = nn.Conv1d, 
                 reinit: bool = True, 
                 win_length: Optional[int] = None,
                 cnn_channels: tuple = (256, 512),
                 dropout: float = 0.2):
        super().__init__()
        
        self.feature_extractor = PANNsFeatureExtractor(
            in_channels=n_channels,
            base_filters=base_filters,
            kernel_sizes=kernel_sizes,
            stride=stride,
            sigmoid=sigmoid,
            output_size=output_size,
            conv=conv,
            reinit=reinit,
            win_length=win_length
        )
        
        self.cnn_head = self._build_cnn_head(
            in_channels=self.feature_extractor.out_chans,
            cnn_channels=cnn_channels
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier1 = nn.Linear(cnn_channels[-1], num_classes)
        self.classifier2 = nn.Linear(cnn_channels[-1], 2)
        
    def _build_cnn_head(self, in_channels: int, cnn_channels: tuple):
        layers = []
        prev_channels = in_channels
        
        for channels in cnn_channels:
            layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((None, None))
            ])
            prev_channels = channels
            
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        x = x.squeeze(1) # (batch_size, seq_len, n_channels)
        x = x.permute(0, 2, 1) # (batch_size, n_channels, seq_len)
        
        features = self.feature_extractor(x) # (batch_size, out_chans, height, time_steps)
        features = self.cnn_head(features) # (batch_size, final_channels, 1, 1)
        features = features.squeeze(-1).squeeze(-1) # (batch_size, final_channels)
        features = self.dropout(features)

        out1 = self.classifier1(features)
        out2 = self.classifier2(features)
        
        return out1, out2