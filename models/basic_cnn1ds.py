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
    def __init__(self, 
                 seq_len=cfg.seq_len,
                 cnn1d_out_channels=32,
                 head_droupout=0.2,
                 attention_n_heads=8,
                 attention_dropout=0.2,
                 num_classes=cfg.main_num_classes):
        super().__init__()
        
        self.channel_sizes = {
            'imu': 3,      # x_imu: 0-2
            'rot': 4,      # x_rot: 3-6  
            'fe1': 13+3,     # x_fe1: 7-19+3
            'fe2': 9,      # x_fe2: 20+3-28+3
            'full': 29+3     # x_full: 0-28+3
        }
        
        self.branch_extractors = nn.ModuleDict()
        
        for branch_name, channel_size in self.channel_sizes.items():
            self.branch_extractors[f'{branch_name}_extractor1'] = Resnet1DFeatureExtractor(
                n_in_channels=channel_size, out_channels=cnn1d_out_channels
            )
            self.branch_extractors[f'{branch_name}_pool1'] = SEPlusMean(cnn1d_out_channels * 4)
            self.branch_extractors[f'{branch_name}_neck1'] = MLPNeck(cnn1d_out_channels * 4)

        final_hidden_dim = (cnn1d_out_channels * 4) * 5
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=cnn1d_out_channels * 4,
            num_heads=attention_n_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(cnn1d_out_channels * 4)

        final_feature_dim = final_hidden_dim * 1

        self.head1 = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_feature_dim // 2, num_classes)
        )

        self.head2 = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_feature_dim // 2, 2)
        )

        self.head3 = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_feature_dim // 2, 4)
        )
            
    def process_extractor(self, x_dict, pad_mask=None):
        branch_features = []
        
        for branch_name in self.channel_sizes.keys():
            x = x_dict[branch_name]
            x_ = x.permute(0, 1, 3, 2).squeeze(1)
            feature = self.branch_extractors[f'{branch_name}_extractor1'](x_)
            feature = self.branch_extractors[f'{branch_name}_pool1'](feature)
            feature = feature + self.branch_extractors[f'{branch_name}_neck1'](feature)
            branch_features.append(feature)
        
        stacked_features = torch.stack(branch_features, dim=1)
        attended_features, _ = self.self_attention(
            stacked_features, 
            stacked_features, 
            stacked_features
        )
        attended_features = self.attention_norm(attended_features + stacked_features)
        final_features = attended_features.view(attended_features.size(0), -1)
        
        return final_features
    
    def forward(self, _x, pad_mask=None):
        # input is (bs, 1, T, C)
        
        x_dict = {
            'imu': _x[:, :, :, :3],
            'rot': _x[:, :, :, 3:7],
            'fe1': _x[:, :, :, 7:20+3],
            'fe2': _x[:, :, :, 20+3:29+3],
            'full': _x
        }
        
        final_features = self.process_extractor(x_dict, pad_mask=pad_mask)

        out1 = self.head1(final_features)
        out2 = self.head2(final_features)
        out3 = self.head3(final_features)

        return out1, out2, out3