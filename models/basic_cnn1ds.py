import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.inceptiontime import InceptionTimeFeatureExtractor
from modules.inceptiontime_replacers import (
    Resnet1DFeatureExtractor, 
    EfficientNet1DFeatureExtractor, 
    DenseNet1DFeatureExtractor
)
from modules.hinception import HInceptionTimeFeatureExtractor
from modules.lite import LiteFeatureExtractor
from configs.config import cfg

# good in hybrid, solo ok

class GeMPool(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool1d(x, kernel_size=x.size(-1)).pow(1.0 / self.p)
        return x.squeeze(-1)

class SEBlock(nn.Module):
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // r, 1), nn.ReLU(),
            nn.Conv1d(channels // r, channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))

class SEPlusMean(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.se = SEBlock(channels)

    def forward(self, x):
        x = self.se(x)
        return x.mean(-1)

class MLPNeck(nn.Module):
    def __init__(self, dim, expansion=2, p=0.2):
        super().__init__()
        self.neck = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(dim * expansion, dim),
            nn.GELU()
        )

    def forward(self, x): # x: (bs, C)
        return self.neck(x)

class CNN1D_SingleSensor_v1(nn.Module):
    def __init__(
        self,
        extractor=cfg.cnn1d_extractor,
        out_channels=cfg.cnn1d_out_channels,
        pooling=cfg.cnn1d_pooling,
        use_neck=cfg.cnn1d_use_neck,
        n_in_channels=cfg.imu_vars,
        num_classes=cfg.main_num_classes,
    ) -> None:
        super().__init__()

        if extractor == 'inception_time':
            self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=n_in_channels, out_channels=out_channels)
        elif extractor == 'resnet':
            self.feature_extractor = Resnet1DFeatureExtractor(n_in_channels=n_in_channels, out_channels=out_channels)
        elif extractor == 'efficientnet':
            self.feature_extractor = EfficientNet1DFeatureExtractor(n_in_channels=n_in_channels, out_channels=out_channels)
        elif extractor == 'densenet':
            self.feature_extractor = DenseNet1DFeatureExtractor(n_in_channels=n_in_channels, out_channels=out_channels)
        elif extractor == 'lite':
            self.feature_extractor = LiteFeatureExtractor(n_in_channels=n_in_channels, out_channels=out_channels)
        elif extractor == 'hinception':
            self.feature_extractor = HInceptionTimeFeatureExtractor(in_channels=n_in_channels, length_TS=cfg.seq_len, n_filters=out_channels, depth=6)

        feature_dim = out_channels * 4

        if pooling == 'gap':
            self.pool = nn.AdaptiveAvgPool1d(1)
            self._pool_forward = lambda t: self.pool(t).squeeze(-1)
        elif pooling == 'gem':
            self.pool = GeMPool()
            self._pool_forward = self.pool
        elif pooling == 'se_mean':
            self.pool = SEPlusMean(feature_dim)
            self._pool_forward = self.pool
        
        self.use_neck = use_neck
        if self.use_neck:
            self.neck = MLPNeck(feature_dim)

        self.head1 = nn.Linear(feature_dim, num_classes)
        self.head2 = nn.Linear(feature_dim, 2)

    def forward(self, x, pad_mask=None):
        # input: (bs, 1, l, c)
        x = x.squeeze(1).transpose(1, 2) # (bs, c, l)
        x = self.feature_extractor(x)
        x = self._pool_forward(x)
        if self.use_neck:
            x = x + self.neck(x)
        y1 = self.head1(x)
        y2 = self.head2(x)
        return y1, y2