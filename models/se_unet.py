import torch
import torch.nn as nn
import torch.nn.functional as F

# https://www.kaggle.com/code/jsday96/parkinsons-overlapping-se-unet-frequency-domain

class Conv1dBlockSE(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, reduction=16, dropout=0):
        super(Conv1dBlockSE, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, stride, padding, bias=False, groups=1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout1d(p=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Squeeze
        se = self.pool(x)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)

        # Excitation
        x = x * se.unsqueeze(2)

        # Add residual and return
        x += residual

        return x

class NonResidualConvSE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, reduction=16, dropout=0):
        super(NonResidualConvSE, self).__init__()

        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p = dropout),
        )

        self.Pooling = nn.AdaptiveAvgPool1d(1)
        self.SqueezeExcitationWeightGenerator = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction),
            nn.ReLU(),
            nn.Linear(out_channels // reduction, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Conv(x)
        pooled_x = self.Pooling(x)
        excitation_weights = self.SqueezeExcitationWeightGenerator(pooled_x.view(pooled_x.size(0), -1))
        x = x * excitation_weights.unsqueeze(2)

        return x

class Conv1dBlockResidual(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dropout=0):
        super(Conv1dBlockResidual, self).__init__()

        self.Layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout1d(p = dropout),
            nn.Conv1d(channels, channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout1d(p = dropout)
        )

    def forward(self, x):
        x = x + self.Layers(x)

        return x

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0):
        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Conv1dBlockPreprocessedSE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, use_second_se=False, preprocessor_dropout=0, se_dropout=0):
        super(Conv1dBlockPreprocessedSE, self).__init__()

        KERNEL_SIZE=3
        STRIDE=1
        PADDING=1
        self.Preprocessor = Conv1dBlock(in_channels, out_channels, KERNEL_SIZE, STRIDE, PADDING, preprocessor_dropout)
        self.SqueezeAndExcitation1 = Conv1dBlockSE(out_channels, KERNEL_SIZE, STRIDE, PADDING, reduction, se_dropout)
        self.SqueezeAndExcitation2 = Conv1dBlockSE(out_channels, KERNEL_SIZE, STRIDE, PADDING, reduction, se_dropout)

        self.UseSecondSe = use_second_se

    def forward(self, x):
        x = self.Preprocessor(x)
        x = self.SqueezeAndExcitation1(x)
        if self.UseSecondSe:
            x = self.SqueezeAndExcitation2(x)
        return x

class SE_Unet_SingleSensor_v1(nn.Module):
    def __init__(
            self, 
            in_channels=7,
            num_classes_target=18,
            num_classes_aux=4, 
            num_classes_aux2=2,
            model_width_coef=32, 
            reduction=16, 
            use_second_se=False, 
            preprocessor_dropout=0, 
            se_dropout=0,
            initial_dropout=0,
            center_dropout=0):
        super(SE_Unet_SingleSensor_v1, self).__init__()

        features = model_width_coef

        self.encoder1 = nn.Sequential(
            NonResidualConvSE(in_channels, features, reduction = reduction//2, dropout=initial_dropout),
            NonResidualConvSE(features, features, reduction = reduction//2, dropout=initial_dropout),
            Conv1dBlockPreprocessedSE(features, features, reduction, use_second_se, preprocessor_dropout, se_dropout)
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = Conv1dBlockPreprocessedSE(features, features*2, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = Conv1dBlockPreprocessedSE(features*2, features*4, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = Conv1dBlockPreprocessedSE(features*4, features*8, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder5 = Conv1dBlockPreprocessedSE(features*8, features*16, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = Conv1dBlock(features*16, features*32, dropout=center_dropout)

        self.upconv5 = nn.ConvTranspose1d(features*32, features*16, kernel_size=2, stride=2)
        self.decoder5 = Conv1dBlockPreprocessedSE(features*32, features*16, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.upconv4 = nn.ConvTranspose1d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = Conv1dBlockPreprocessedSE(features*16, features*8, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.upconv3 = nn.ConvTranspose1d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = Conv1dBlockPreprocessedSE(features*8, features*4, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.upconv2 = nn.ConvTranspose1d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = Conv1dBlockPreprocessedSE(features*4, features*2, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.upconv1 = nn.ConvTranspose1d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = Conv1dBlockPreprocessedSE(features*2, features, reduction, use_second_se, preprocessor_dropout, se_dropout)

        self.feature_conv = nn.Conv1d(features, features, kernel_size=1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier_target = nn.Sequential(
            nn.Linear(features * 2, features),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(features, features//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(features//2, num_classes_target)
        )
        
        self.classifier_aux = nn.Sequential(
            nn.Linear(features * 2, features//2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(features//2, num_classes_aux)
        )
        
        self.classifier_aux2 = nn.Sequential(
            nn.Linear(features * 2, features//4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(features//4, num_classes_aux2)
        )

    def forward(self, imu_data):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
        Returns:
            dict with 'target', 'aux_target', 'aux2_target' logits
        """
        batch_size = imu_data.shape[0]
        x = imu_data.squeeze(1)  # [B, L, 7]
        x = x.permute(0, 2, 1)   # [B, 7, L]

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.upconv5(bottleneck)
        dec5 = self.decoder5(torch.cat((dec5, enc5), dim=1))
        dec4 = self.upconv4(dec5)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        temporal_features = self.feature_conv(dec1)  # [B, features, L]
        
        avg_pooled = self.global_avg_pool(temporal_features)  # [B, features, 1]
        max_pooled = self.global_max_pool(temporal_features)  # [B, features, 1]
        
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)  # [B, features*2, 1]
        features = pooled_features.squeeze(-1)  # [B, features*2]
        
        target_logits = self.classifier_target(features)      # [B, 18]
        aux_logits = self.classifier_aux(features)            # [B, 4]
        aux2_logits = self.classifier_aux2(features)          # [B, 2]
        
        return target_logits, aux_logits, aux2_logits
    
class MultiSensorAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: [B, num_sensors, embed_dim]
        attn_out, _ = self.multihead_attn(x, x, x)
        return self.norm(x + attn_out)

class SensorFusionModule(nn.Module):
    def __init__(self, imu_dim, tof_dim, thm_dim, fusion_dim):
        super().__init__()
        
        self.imu_proj = nn.Linear(imu_dim, fusion_dim)
        self.tof_proj = nn.Linear(tof_dim, fusion_dim)
        self.thm_proj = nn.Linear(thm_dim, fusion_dim)
        
        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU()
        )
        
    def forward(self, imu_features, tof_features, thm_features):
        imu_proj = self.imu_proj(imu_features)  # [B, fusion_dim]
        tof_proj = self.tof_proj(tof_features)  # [B, fusion_dim]
        thm_proj = self.thm_proj(thm_features)  # [B, fusion_dim]
        
        sensor_seq = torch.stack([imu_proj, tof_proj, thm_proj], dim=1)
        
        attended_features, _ = self.cross_attention(sensor_seq, sensor_seq, sensor_seq)
        
        fused = torch.cat([
            attended_features[:, 0],  # IMU
            attended_features[:, 1],  # TOF  
            attended_features[:, 2]   # THM
        ], dim=1)
        
        return self.fusion_layers(fused)

class MultiSensor_SE_Unet_v1(nn.Module):
    def __init__(
            self, 
            model_width_coef=32, 
            reduction=16, 
            use_second_se=False, 
            preprocessor_dropout=0, 
            se_dropout=0,
            initial_dropout=0,
            center_dropout=0,
            num_classes_target=18,
            num_classes_aux=4, 
            num_classes_aux2=2):
        super(MultiSensor_SE_Unet_v1, self).__init__()

        features = model_width_coef
        
        self.imu_encoder1 = nn.Sequential(
            NonResidualConvSE(7, features, reduction=reduction//2, dropout=initial_dropout),
            NonResidualConvSE(features, features, reduction=reduction//2, dropout=initial_dropout),
            Conv1dBlockPreprocessedSE(features, features, reduction, use_second_se, preprocessor_dropout, se_dropout)
        )
        self.imu_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.imu_encoder2 = Conv1dBlockPreprocessedSE(features, features*2, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.imu_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.imu_encoder3 = Conv1dBlockPreprocessedSE(features*2, features*4, reduction, use_second_se, preprocessor_dropout, se_dropout)
        
        self.tof_single_encoder1 = nn.Sequential(
            NonResidualConvSE(64, features, reduction=reduction//2, dropout=initial_dropout),
            NonResidualConvSE(features, features, reduction=reduction//2, dropout=initial_dropout),
            Conv1dBlockPreprocessedSE(features, features, reduction, use_second_se, preprocessor_dropout, se_dropout)
        )
        self.tof_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.tof_single_encoder2 = Conv1dBlockPreprocessedSE(features, features*2, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.tof_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.tof_single_encoder3 = Conv1dBlockPreprocessedSE(features*2, features*4, reduction, use_second_se, preprocessor_dropout, se_dropout)
        
        self.tof_sensor_attention = MultiSensorAttention(features*4, num_heads=4)
        
        self.thm_single_encoder1 = nn.Sequential(
            NonResidualConvSE(1, features//2, reduction=reduction//4, dropout=initial_dropout),
            NonResidualConvSE(features//2, features//2, reduction=reduction//4, dropout=initial_dropout),
            Conv1dBlockPreprocessedSE(features//2, features//2, reduction, use_second_se, preprocessor_dropout, se_dropout)
        )
        self.thm_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.thm_single_encoder2 = Conv1dBlockPreprocessedSE(features//2, features, reduction, use_second_se, preprocessor_dropout, se_dropout)
        self.thm_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.thm_single_encoder3 = Conv1dBlockPreprocessedSE(features, features*2, reduction, use_second_se, preprocessor_dropout, se_dropout)
        
        self.thm_sensor_attention = MultiSensorAttention(features*2, num_heads=4)
        
        self.imu_global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.imu_global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.tof_global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.tof_global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.thm_global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.thm_global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        fusion_dim = features * 4
        self.sensor_fusion = SensorFusionModule(
            imu_dim=features*4*2,
            tof_dim=features*4*2,  
            thm_dim=features*2*2,
            fusion_dim=fusion_dim
        )
        
        self.classifier_target = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim//2, fusion_dim//4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim//4, num_classes_target)
        )
        
        self.classifier_aux = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim//4, num_classes_aux)
        )
        
        self.classifier_aux2 = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim//8, num_classes_aux2)
        )

    def process_single_sensor_stack(self, sensor_data, encoder1, pool1, encoder2, pool2, encoder3, avg_pool, max_pool):
        batch_size, num_sensors, seq_len, features = sensor_data.shape
        
        sensor_features = []
        for i in range(num_sensors):
            single_sensor = sensor_data[:, i, :, :].permute(0, 2, 1)  # [B, features, L]
            
            enc1 = encoder1(single_sensor)
            enc2 = encoder2(pool1(enc1))
            enc3 = encoder3(pool2(enc2))
            
            avg_pooled = avg_pool(enc3)  # [B, channels, 1]
            max_pooled = max_pool(enc3)  # [B, channels, 1]
            pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # [B, channels*2, 1]
            pooled = pooled.squeeze(-1)  # [B, channels*2]
            
            sensor_features.append(pooled)
        
        stacked_features = torch.stack(sensor_features, dim=1)
        return stacked_features

    def forward(self, imu_data, thm_data, tof_data):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data  
            thm_data: [B, 5, L, 1] - Thermal sensor data
        Returns:
            tuple of (target_logits, aux_logits, aux2_logits)
        """
        
        imu_x = imu_data.squeeze(1).permute(0, 2, 1)  # [B, 7, L]
        
        imu_enc1 = self.imu_encoder1(imu_x)
        imu_enc2 = self.imu_encoder2(self.imu_pool1(imu_enc1))
        imu_enc3 = self.imu_encoder3(self.imu_pool2(imu_enc2))
        
        imu_avg_pooled = self.imu_global_avg_pool(imu_enc3)
        imu_max_pooled = self.imu_global_max_pool(imu_enc3)
        imu_features = torch.cat([imu_avg_pooled, imu_max_pooled], dim=1).squeeze(-1)
        
        tof_sensor_features = self.process_single_sensor_stack(
            tof_data, self.tof_single_encoder1, self.tof_pool1, 
            self.tof_single_encoder2, self.tof_pool2, self.tof_single_encoder3,
            self.tof_global_avg_pool, self.tof_global_max_pool
        )
        
        tof_attended = self.tof_sensor_attention(tof_sensor_features)  # [B, 5, feature_dim]
        tof_features = tof_attended.mean(dim=1)  # [B, feature_dim]
        
        thm_sensor_features = self.process_single_sensor_stack(
            thm_data, self.thm_single_encoder1, self.thm_pool1,
            self.thm_single_encoder2, self.thm_pool2, self.thm_single_encoder3,
            self.thm_global_avg_pool, self.thm_global_max_pool
        )
        
        thm_attended = self.thm_sensor_attention(thm_sensor_features)  # [B, 5, feature_dim]
        thm_features = thm_attended.mean(dim=1)  # [B, feature_dim]
        
        fused_features = self.sensor_fusion(imu_features, tof_features, thm_features)
        
        target_logits = self.classifier_target(fused_features)
        aux_logits = self.classifier_aux(fused_features)
        aux2_logits = self.classifier_aux2(fused_features)
        
        return target_logits, aux_logits, aux2_logits