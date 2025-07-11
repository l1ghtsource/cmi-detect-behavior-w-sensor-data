import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

from configs.config import cfg

# bad solo, bad in hybrid :(

class IMUNet_SingleSensor_v1(nn.Module):
    def __init__(
        self,
        in_channels=1,
        imu_channels=cfg.imu_vars,
        seq_len=cfg.seq_len,
        num_classes=18,
        n_temporal_filters=16,
        kernel_size_temporal=10,
        target_size=256,
        p_dropout=0.2,
    ):
        super().__init__()
        
        self.pad_temporal = nn.ZeroPad2d(
            (kernel_size_temporal // 2, kernel_size_temporal // 2, 0, 0)
        )
        self.conv1d_temporal = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_temporal_filters,
            kernel_size=(1, kernel_size_temporal),
            padding=0
        )
        self.bn_temporal = nn.BatchNorm2d(n_temporal_filters)
        self.silu = nn.SiLU()
        
        self.target_size = target_size
        self.n_temporal_filters = n_temporal_filters
        self.imu_channels = imu_channels
        
        # One timm model only
        self.cnn = timm.create_model(
            'swinv2_tiny_window16_256',
            pretrained=True,
            drop_rate=p_dropout,
            in_chans=1,
            num_classes=0,
        )
        
        self.gru = nn.GRU(
            input_size=target_size,
            hidden_size=256,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        
        self._get_feature_dims()
        
        total_features = self.cnn_features + self.gru_features
        self.classifier1 = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(total_features, num_classes)
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(total_features, 2)
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(total_features, 4)
        )

    def _get_feature_dims(self):
        with torch.no_grad():
            test_input = torch.randn(1, 1, self.target_size, self.target_size)
            features = self.cnn(test_input)
            self.cnn_features = features.shape[1]
            self.gru_features = 256

    def create_imu_image(self, imu_data):
        return imu_data.permute(0, 2, 1).unsqueeze(1)

    def resize_to_target(self, x):
        return F.interpolate(x, size=(self.target_size, self.target_size), 
                           mode='bilinear', align_corners=False)

    def forward(self, imu_data, pad_mask=None):
        if imu_data.dim() == 4:
            imu_data = imu_data.squeeze(1)  # (bs, seq_len, channels)
        
        x = self.create_imu_image(imu_data)  # (bs, 1, imu_channels, seq_len)
        
        # Temporal convolution
        x = self.pad_temporal(x)
        x = self.conv1d_temporal(x)  # (bs, n_temporal_filters, imu_channels, seq_len)
        x = self.bn_temporal(x)
        x = self.silu(x)
        
        # Flatten filters and channels in natural order
        # Shape: (bs, 1, n_temporal_filters * imu_channels, seq_len)
        x_stacked = x.view(x.size(0), 1, self.n_temporal_filters * self.imu_channels, x.size(3))
        
        # Resize for CNN
        x_resized = self.resize_to_target(x_stacked)  # (bs, 1, target_size, target_size)
        cnn_out = self.cnn(x_resized)
        
        # GRU
        gru_input = x_resized.squeeze(1).permute(0, 2, 1)  # (bs, target_size, target_size)
        gru_out, _ = self.gru(gru_input)
        gru_out = gru_out[:, -1, :]  # (bs, 256)
        
        # Classification
        combined_features = torch.cat([cnn_out, gru_out], dim=1)
        
        output1 = self.classifier1(combined_features)
        output2 = self.classifier2(combined_features)
        output3 = self.classifier3(combined_features)
        
        return output1, output2, output3