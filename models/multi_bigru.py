import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import cfg

# good in hybrid, solo ok

# https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/discussion/416410

class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, new_h
    
class MultiResidualBiGRU_SingleSensor_Extractor(nn.Module):
    def __init__(self, 
                 seq_len=cfg.seq_len,
                 n_imu_vars=cfg.imu_vars,
                 hidden_size=128, 
                 n_layers=3, 
                 bidir=True,
                 dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.n_imu_vars = n_imu_vars
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.imu_input_projection = nn.Linear(n_imu_vars, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        self.res_bigrus = nn.ModuleList([
            ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir)
            for _ in range(n_layers)
        ])
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        pooled_size = hidden_size * 2
        self.feature_fusion = nn.Sequential(
            nn.Linear(pooled_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, imu_data, pad_mask=None, h=None):
        batch_size = imu_data.size(0)
        
        x = imu_data.squeeze(1)
        
        x = self.imu_input_projection(x)  # (B, L, hidden_size)
        x = self.input_ln(x)
        x = nn.functional.relu(x)
        x = self.input_dropout(x)
        
        if h is None:
            h = [None for _ in range(self.n_layers)]
        
        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)
        
        # x shape: (B, L, hidden_size) -> (B, hidden_size, L) for pooling
        x_transposed = x.transpose(1, 2)
        
        avg_pooled = self.global_avg_pool(x_transposed).squeeze(-1)  # (B, hidden_size)
        max_pooled = self.global_max_pool(x_transposed).squeeze(-1)  # (B, hidden_size)
        
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)  # (B, hidden_size*2)
        features = self.feature_fusion(pooled_features)  # (B, hidden_size)
        
        return features

class MultiResidualBiGRU_SingleSensor_v1(nn.Module):
    def __init__(self, 
                 multibigru_dim=64,#128, -> 64 the same
                 multibigru_layers=3,# 3 -> ok
                 multibigru_dropout=0.3,#0.2, -> 0.3 better
                 seq_len=cfg.seq_len,
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
            self.branch_extractors[f'{branch_name}_extractor1'] = MultiResidualBiGRU_SingleSensor_Extractor(
                seq_len=seq_len,
                n_imu_vars=channel_size,
                hidden_size=multibigru_dim, 
                n_layers=multibigru_layers, 
                bidir=True,
                dropout=multibigru_dropout
            )

        final_hidden_dim = (multibigru_dim) * 5
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=multibigru_dim,
            num_heads=attention_n_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(multibigru_dim)

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
            feature = self.branch_extractors[f'{branch_name}_extractor1'](x)
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