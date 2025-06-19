import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import cfg

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
    
class MultiResidualBiGRU_SingleSensor_v1(nn.Module):
    def __init__(self, 
                 seq_len=cfg.seq_len,
                 n_imu_vars=7,
                 hidden_size=128, 
                 n_layers=3, 
                 bidir=True,
                 num_classes=18,
                 dropout=0.1):
        super(MultiResidualBiGRU_SingleSensor_v1, self).__init__()

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
        
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.target_head2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, imu_data, pad_mask=None):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
        Returns:
            Dictionary with predictions for all three tasks
        """
        batch_size = imu_data.size(0)
        
        x = imu_data.squeeze(1)  # Remove sensor dimension
        
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
        
        target_logits = self.target_head(features)
        target_logits2 = self.target_head2(features)
        
        return target_logits, target_logits2
    
class SensorProcessor(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, bidir=True, dropout=0.1):
        super(SensorProcessor, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        self.res_bigrus = nn.ModuleList([
            ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, h=None):
        """
        Args:
            x: (B, L, input_size)
        Returns:
            processed features: (B, L, hidden_size)
        """
        x = self.input_projection(x)
        x = self.input_ln(x)
        x = F.relu(x)
        x = self.input_dropout(x)
        
        if h is None:
            h = [None for _ in range(len(self.res_bigrus))]
        
        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)
            
        return x, new_h

class IntraSensorAttention(nn.Module):
    def __init__(self, hidden_size, num_sensors, dropout=0.1):
        super(IntraSensorAttention, self).__init__()
        
        self.num_sensors = num_sensors
        self.hidden_size = hidden_size
        
        self.sensor_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sensor_features):
        """
        Args:
            sensor_features: (B, num_sensors, L, hidden_size)
        Returns:
            attended_features: (B, num_sensors, L, hidden_size)
        """
        B, num_sensors, L, hidden_size = sensor_features.shape
        
        x = sensor_features.transpose(1, 2).contiguous()  # (B, L, num_sensors, hidden_size)
        x = x.view(B * L, num_sensors, hidden_size)
        
        attended, _ = self.sensor_attention(x, x, x)
        attended = self.dropout(attended)
        
        attended = self.norm(attended + x)
        
        attended = attended.view(B, L, num_sensors, hidden_size)
        attended = attended.transpose(1, 2).contiguous()
        
        return attended

class CrossSensorFusion(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(CrossSensorFusion, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, query_features, key_value_features):
        """
        Args:
            query_features: (B, L, hidden_size)
            key_value_features: (B, L, hidden_size)
        """
        attended, _ = self.cross_attention(
            query_features, key_value_features, key_value_features
        )
        
        x = self.norm1(attended + query_features)
        ffn_out = self.ffn(x)
        x = self.norm2(ffn_out + x)
        
        return x

class MultiSensor_MultiResidualBiGRU_v1(nn.Module):
    def __init__(self, 
                 seq_len=cfg.seq_len,
                 hidden_size=128, 
                 n_layers=2, 
                 bidir=True,
                 num_classes=18,
                 dropout=0.1):
        super(MultiSensor_MultiResidualBiGRU_v1, self).__init__()

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.imu_processor = SensorProcessor(
            input_size=7, hidden_size=hidden_size, 
            n_layers=n_layers, bidir=bidir, dropout=dropout
        )
        
        self.tof_processor = SensorProcessor(
            input_size=64, hidden_size=hidden_size, 
            n_layers=n_layers, bidir=bidir, dropout=dropout
        )
        
        self.thm_processor = SensorProcessor(
            input_size=1, hidden_size=hidden_size, 
            n_layers=n_layers, bidir=bidir, dropout=dropout
        )
        
        self.tof_intra_attention = IntraSensorAttention(
            hidden_size=hidden_size, num_sensors=5, dropout=dropout
        )
        
        self.thm_intra_attention = IntraSensorAttention(
            hidden_size=hidden_size, num_sensors=5, dropout=dropout
        )
        
        self.imu_tof_fusion = CrossSensorFusion(hidden_size, dropout)
        self.imu_thm_fusion = CrossSensorFusion(hidden_size, dropout)
        self.tof_thm_fusion = CrossSensorFusion(hidden_size, dropout)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        fused_size = hidden_size * 6
        self.feature_fusion = nn.Sequential(
            nn.Linear(fused_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self.target_head2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, imu_data, thm_data, tof_data, pad_mask=None):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data  
            thm_data: [B, 5, L, 1] - Thermal sensor data
        """
        batch_size = imu_data.size(0)
        
        imu_x = imu_data.squeeze(1)  # (B, L, 7)
        imu_features, _ = self.imu_processor(imu_x)  # (B, L, hidden_size)
        
        tof_features_list = []
        for i in range(5):
            tof_sensor_data = tof_data[:, i, :, :]  # (B, L, 64)
            tof_sensor_features, _ = self.tof_processor(tof_sensor_data)  # (B, L, hidden_size)
            tof_features_list.append(tof_sensor_features)
        
        tof_features = torch.stack(tof_features_list, dim=1)  # (B, 5, L, hidden_size)
        
        thm_features_list = []
        for i in range(5):
            thm_sensor_data = thm_data[:, i, :, :]  # (B, L, 1)
            thm_sensor_features, _ = self.thm_processor(thm_sensor_data)  # (B, L, hidden_size)
            thm_features_list.append(thm_sensor_features)
        
        thm_features = torch.stack(thm_features_list, dim=1)  # (B, 5, L, hidden_size)
        
        tof_attended = self.tof_intra_attention(tof_features)  # (B, 5, L, hidden_size)
        tof_aggregated = tof_attended.mean(dim=1)  # (B, L, hidden_size)
        
        thm_attended = self.thm_intra_attention(thm_features)  # (B, 5, L, hidden_size)
        thm_aggregated = thm_attended.mean(dim=1)  # (B, L, hidden_size)
        
        imu_tof_fused = self.imu_tof_fusion(imu_features, tof_aggregated)
        imu_thm_fused = self.imu_thm_fusion(imu_features, thm_aggregated)
        tof_thm_fused = self.tof_thm_fusion(tof_aggregated, thm_aggregated)
        
        def apply_pooling(features):
            # (B, L, hidden_size) -> (B, hidden_size, L) -> (B, hidden_size)
            features_t = features.transpose(1, 2)
            avg_pool = self.global_avg_pool(features_t).squeeze(-1)
            max_pool = self.global_max_pool(features_t).squeeze(-1)
            return torch.cat([avg_pool, max_pool], dim=1)  # (B, hidden_size*2)
        
        imu_tof_pooled = apply_pooling(imu_tof_fused)
        imu_thm_pooled = apply_pooling(imu_thm_fused)  
        tof_thm_pooled = apply_pooling(tof_thm_fused)
        
        all_features = torch.cat([
            imu_tof_pooled, imu_thm_pooled, tof_thm_pooled
        ], dim=1)  # (B, hidden_size*6)
        
        fused_features = self.feature_fusion(all_features)  # (B, hidden_size)
        
        target_logits = self.target_head(fused_features)
        target_logits2 = self.target_head2(fused_features)
        
        return target_logits, target_logits2