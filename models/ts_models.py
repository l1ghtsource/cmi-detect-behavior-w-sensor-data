import torch
import torch.nn as nn
from modules.ts_modules import DilatedCNN, SensorAttn

class TS_SimpleMSModel(nn.Module):
    def __init__(self, imu_features, thm_features, tof_features, num_classes, hidden_dim=64, dropout=0.2):
        super().__init__()
        
        self.imu_lstm = nn.LSTM(imu_features, hidden_dim, batch_first=True, bidirectional=True)
        self.imu_dropout = nn.Dropout(dropout)
        
        self.thm_lstm = nn.LSTM(thm_features, hidden_dim, batch_first=True, bidirectional=True)
        self.thm_dropout = nn.Dropout(dropout)
        
        self.tof_lstm = nn.LSTM(tof_features, hidden_dim, batch_first=True, bidirectional=True)
        self.tof_dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim * 6, num_classes)
    
    def forward(self, imu, thm, tof):
        imu_out, _ = self.imu_lstm(imu)
        imu_out = self.imu_dropout(imu_out)
        
        thm_out, _ = self.thm_lstm(thm)
        thm_out = self.thm_dropout(thm_out)
        
        tof_out, _ = self.tof_lstm(tof)
        tof_out = self.tof_dropout(tof_out)
        
        combined_features = torch.cat((imu_out[:, -1, :], thm_out[:, -1, :], tof_out[:, -1, :]), dim=1)
        
        output = self.fc(combined_features)
        return output

class TS_MSModel(nn.Module):
    def __init__(self, imu_features, thm_features, tof_features, num_classes, hidden_dim=128):
        super().__init__()
        
        self.imu_cnn = nn.Sequential(
            DilatedCNN(imu_features, 64, 1),
            DilatedCNN(64, hidden_dim, 2)
        )
        self.imu_lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.imu_attn = SensorAttn(hidden_dim * 2)

        self.thm_cnn = nn.Sequential(
            DilatedCNN(thm_features, 64, 1),
            DilatedCNN(64, hidden_dim, 3)
        )
        self.thm_lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.thm_attn = SensorAttn(hidden_dim * 2)

        self.tof_cnn = nn.Sequential(
            DilatedCNN(tof_features, 64, 1),
            DilatedCNN(64, hidden_dim, 4)
        )
        self.tof_lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.tof_attn = SensorAttn(hidden_dim * 2)

        self.cross_attn = nn.MultiheadAttention(hidden_dim * 6, 8, batch_first=True)
        self.temporal_attn = nn.TransformerEncoderLayer(hidden_dim * 6, 8, dropout=0.2, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*6 * 2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, imu, thm, tof):
        imu = self.imu_cnn(imu.permute(0, 2, 1)).permute(0, 2, 1)
        imu, _ = self.imu_lstm(imu)
        imu = self.imu_attn(imu)
        
        thm = self.thm_cnn(thm.permute(0, 2, 1)).permute(0, 2, 1)
        thm, _ = self.thm_lstm(thm)
        thm = self.thm_attn(thm)
        
        tof = self.tof_cnn(tof.permute(0, 2 ,1)).permute(0, 2, 1)
        tof, _ = self.tof_lstm(tof)
        tof = self.tof_attn(tof)

        combined = torch.cat([imu, thm, tof], dim=2)
        attn_out, _ = self.cross_attn(combined, combined, combined)
        temporal_out = self.temporal_attn(attn_out)
        
        avg_pool = temporal_out.mean(dim=1)
        max_pool = temporal_out.max(dim=1).values
        features = torch.cat([avg_pool, max_pool], dim=1)
        
        return self.fc(features)