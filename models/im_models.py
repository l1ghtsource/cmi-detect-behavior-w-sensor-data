import torch
import torch.nn as nn
import timm
from configs.config import cfg

# TODO: test it

class IMG_CMIModel(nn.Module):
    def __init__(
        self,
        encoder_name=cfg.encoder_name,
        num_classes=cfg.main_num_classes,
        pretrained=cfg.pretrained,
        num_attention_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        
        self.imu_encoder = self._create_encoder(encoder_name, pretrained)
        self.thm_encoder = self._create_encoder(encoder_name, pretrained)
        self.tof_encoder = self._create_encoder(encoder_name, pretrained)
        
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=cfg.encoder_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(cfg.encoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(cfg.encoder_hidden_dim * 3, cfg.encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.encoder_hidden_dim, num_classes)
        )
        
    def _create_encoder(self, encoder_name, pretrained):
        model = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        return model

    def forward(self, imu_images, thm_images, tof_images):
        # inputs: (batch_size, 3, H, W)

        imu_features = self.imu_encoder(imu_images) # (batch_size, encoder_dim)
        thm_features = self.thm_encoder(thm_images) # (batch_size, encoder_dim)
        tof_features = self.tof_encoder(tof_images) # (batch_size, encoder_dim)
        
        feature_sequence = torch.stack([imu_features, thm_features, tof_features], dim=1) # (batch_size, 3, encoder_dim)

        attended_features, _ = self.multihead_attention(feature_sequence, feature_sequence, feature_sequence) # (batch_size, 3, encoder_dim)
        
        attended_features = self.layer_norm(attended_features) # (batch_size, 3, encoder_dim)
        attended_features = self.dropout(attended_features) # (batch_size, 3, encoder_dim)
        
        combined_features = attended_features.flatten(start_dim=1) # (batch_size, 3 * encoder_dim)
        
        logits = self.classifier(combined_features) # (batch_size, num_classes)
        
        return logits