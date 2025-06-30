import torch.nn as nn
import timm
from modules.spectrogram import SpecFeatureExtractor
from configs.config import cfg

# TODO: test it
    
class IMG_CMIModel(nn.Module):
    def __init__(self, n_classes=cfg.main_num_classes, dropout=cfg.timemil_dropout,
                 spec_height=64, spec_hop_length=32, spec_win_length=None):
        super().__init__()
        
        self.spec_feature_extractor = SpecFeatureExtractor(
            in_channels=cfg.imu_vars,
            height=spec_height,
            hop_length=spec_hop_length,
            win_length=spec_win_length,
            out_size=224
        )
        
        self.channel_adapter = nn.Conv2d(
            in_channels=cfg.imu_vars,
            out_channels=3,
            kernel_size=1,
            bias=False
        )
        
        self.backbone = timm.create_model(cfg.encoder_name, pretrained=True)
        
        feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self._fc_main = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, n_classes)
        )
        
        self._fc_seq_type = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 2)
        )
        
        self._initialize_custom_layers()
        
    def _initialize_custom_layers(self):
        """Initialize weights for custom layers"""
        nn.init.xavier_uniform_(self.channel_adapter.weight)
        
        for module in [self._fc_main, self._fc_seq_type]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, imu_data, pad_mask=None, warmup=False):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            pad_mask: [B, L] - padding mask (optional)
            warmup: bool - whether to use warmup strategy (optional)
        """
        B, _, L, _ = imu_data.shape
        
        sensor_data = imu_data[:, 0, :, :].transpose(1, 2)  # [B, 7, L]
        spec_features = self.spec_feature_extractor(sensor_data)  # [B, 7, 224, 224]
        spec_features = self.channel_adapter(spec_features)  # [B, 3, 224, 224]
        features = self.backbone(spec_features)  # [B, 1280]
        
        logits_main = self._fc_main(features)
        logits_seq_type = self._fc_seq_type(features)
        
        return logits_main, logits_seq_type