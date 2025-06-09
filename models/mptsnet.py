import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.mpts_layers import DataEmbedding, clsWindowTransformer, Inception_CBAM, clsTransformer, DataEmbedding_v1, Transformer, WindowTransformer

def fft_find_each_amplitude(data, target_period):
    '''
    For each element in a batch
    :param data:
    :param target_period:
    :return: target amplitudes
    '''
    batch_size = data.shape[0]
    sequence_length = data.shape[1]
    T = 1.0 / sequence_length  # sampling interval

    # Initialize an array to store the amplitude for each batch element
    amplitudes = torch.zeros((batch_size, 1))

    for i in range(batch_size):
        # For each batch element, average the num_channels dimension
        averaged_data = data[i].mean(axis=-1)

        # Compute FFT
        yf = np.fft.fft(averaged_data)
        xf = np.fft.fftfreq(sequence_length, T)[:sequence_length // 2]
        power_spectrum = 2.0 / sequence_length * np.abs(yf[:sequence_length // 2])

        # Calculate the target frequency
        target_frequency = sequence_length / target_period
        # Find the closest frequency index
        closest_index = np.argmin(np.abs(xf - target_frequency))
        closest_amplitude = power_spectrum[closest_index]

        # Store the amplitude of the current batch element
        amplitudes[i] = closest_amplitude

    return amplitudes

class PeriodicBlock(nn.Module):
    def __init__(self, flag, periods, seq_length, embed_dim, embed_dim_t, num_heads, ff_dim, num_layers):
        super(PeriodicBlock, self).__init__()
        self.periods = periods
        self.embed_dim = embed_dim

        self.cnn = nn.Sequential(
            Inception_CBAM(embed_dim, 1024),
            nn.GELU(),
            Inception_CBAM(1024, embed_dim)
        )

        if flag:
            self.transformer = Transformer(embed_dim, seq_length, embed_dim, num_heads, ff_dim, num_layers)

            self.transformers = nn.ModuleList([
                WindowTransformer(embed_dim * period, (seq_length // period) + 1, embed_dim_t, num_heads, ff_dim,
                                  num_layers)
                for period in periods
            ])
        else:
            self.transformer = clsTransformer(seq_length, embed_dim, num_heads, ff_dim, num_layers)

            self.transformers = nn.ModuleList([
                clsWindowTransformer(embed_dim * period, (seq_length // period) + 1, embed_dim_t, num_heads, ff_dim,
                                     num_layers)
                for period in periods
            ])

    def forward(self, x):  # (batch_size, embed_dim, seq_length)
        B = x.shape[0]
        C = x.shape[1]
        T = x.shape[2]

        time_point_features = self.transformer(x)  # (batch_size, embed_dim, T)

        global_features = []
        amplitudes = []
        for i, period in enumerate(self.periods):
            x_fft = x.permute(0, 2, 1).detach().cpu().numpy()
            amplitudes.append(fft_find_each_amplitude(x_fft, period))
            if T % period != 0:
                # padding
                length = ((T // period) + 1) * period
                padding = torch.zeros([B, C, (length - T)]).to(x.device)
                out = torch.cat([x, padding], dim=2)  # (batch_size, embed_dim, length)
            else:
                length = T
                out = x  # (batch_size, embed_dim, T(length))
            num_period = length // period
            # reshape
            out = out.reshape(B, C, period, num_period).contiguous()  # (batch_size, embed_dim, period, num_period)
            local_features = []
            for j in range(num_period):
                feature = self.cnn(out[:, :, :, j])  # (batch_size, embed_dim, period)
                local_features.append(feature)
            local_features = torch.stack(local_features, dim=-1)  # (batch_size, embed_dim, period, num_period)

            # add res part
            local_features = out+local_features

            local_features = local_features.reshape(B, -1, num_period)  # (batch_size, embed_dim*period, num_period)
            global_feature = self.transformers[i](local_features)  # (batch_size, embed_dim*period, num_period)
            global_feature = global_feature.reshape(B, self.embed_dim, -1).contiguous()  # (batch_size, embed_dim, length)
            global_feature = global_feature[:, :, :T]  # (batch_size, embed_dim, T)
            global_features.append(global_feature)

        # Features fusion
        amplitudes = torch.cat(amplitudes, dim=1)

        global_features = torch.stack(global_features, dim=-1)  # (B, embed_dim, T, k)
        weights = torch.softmax(amplitudes, dim=1)  # (B, k)
        period_weight = weights.unsqueeze(1).unsqueeze(1).repeat(1, self.embed_dim, T, 1).to(x.device)  # (B, embed_dim, T, k)
        res = torch.sum(global_features * period_weight, -1)  # (B, embed_dim, T)

        res = res+time_point_features+x

        return res


class Original_MPTSNet(nn.Module):
    def __init__(self, periods, flag, num_channels, seq_length, num_classes, embed_dim, embed_dim_t, num_heads, ff_dim, num_layers):
        super(Original_MPTSNet, self).__init__()
        if flag:
            print('[INFO] True')
            self.enc_embedding = DataEmbedding_v1(num_channels, embed_dim, dropout=0.1)
        else:
            self.enc_embedding = DataEmbedding(num_channels, embed_dim, seq_length, dropout=0.1)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.model = nn.ModuleList([PeriodicBlock(flag, periods, seq_length, embed_dim,
                                    embed_dim_t, num_heads, ff_dim, num_layers)
                                    for _ in range(2)])

        self.activation = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(seq_length * embed_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, num_channels)
        x = self.enc_embedding(x)  # (batch_size, seq_length, embed_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, seq_length)
        for i in range(2):
            x = self.layer_norm(self.model[i](x).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)  # (B, embed_dim * T)
        output = self.fc(x.float())

        return output

# imu only model
class MPTSNet_SingleSensor_v1(nn.Module):
    def __init__(self, periods, flag, num_channels, seq_length, num_classes, embed_dim, embed_dim_t, num_heads, ff_dim, num_layers):
        super(MPTSNet_SingleSensor_v1, self).__init__()
        if flag:
            print('[INFO] True')
            self.enc_embedding = DataEmbedding_v1(num_channels, embed_dim, dropout=0.1)
        else:
            self.enc_embedding = DataEmbedding(num_channels, embed_dim, seq_length, dropout=0.1)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.model = nn.ModuleList([PeriodicBlock(flag, periods, seq_length, embed_dim,
                                    embed_dim_t, num_heads, ff_dim, num_layers)
                                    for _ in range(2)])

        self.activation = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(seq_length * embed_dim, num_classes)

    def forward(self, imu_data):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
        Returns:
            logits: [B, num_classes] - Classification logits
        """
        # Reshape imu_data from [B, 1, L, 7] to [B, 7, L]
        x = imu_data.squeeze(1)  # Remove singleton dimension: [B, 1, L, 7] -> [B, L, 7]
        x = x.permute(0, 2, 1)   # Permute to match expected format: [B, L, 7] -> [B, 7, L]
        
        # Rest of the forward pass remains the same
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, num_channels)
        x = self.enc_embedding(x)  # (batch_size, seq_length, embed_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, seq_length)
        
        for i in range(2):
            x = self.layer_norm(self.model[i](x).permute(0, 2, 1)).permute(0, 2, 1)
        
        x = self.activation(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)  # (B, embed_dim * T)
        output = self.fc(x.float())

        return output

# imu + tof + thm
class MultiSensor_MPTSNet_v1(nn.Module):
    def __init__(self, periods, flag, seq_length, num_classes, embed_dim, embed_dim_t, num_heads, ff_dim, num_layers):
        super(MultiSensor_MPTSNet_v1, self).__init__()
        
        # Separate embedding layers for each sensor type
        if flag:
            self.imu_embedding = DataEmbedding_v1(7, embed_dim, dropout=0.1)  # 7 IMU channels
            self.tof_embedding = DataEmbedding_v1(64, embed_dim, dropout=0.1)  # 64 ToF channels  
            self.thm_embedding = DataEmbedding_v1(1, embed_dim, dropout=0.1)  # 1 thermal channel
        else:
            self.imu_embedding = DataEmbedding(7, embed_dim, seq_length, dropout=0.1)
            self.tof_embedding = DataEmbedding(64, embed_dim, seq_length, dropout=0.1)
            self.thm_embedding = DataEmbedding(1, embed_dim, seq_length, dropout=0.1)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Separate processing blocks for each sensor type
        self.imu_blocks = nn.ModuleList([
            PeriodicBlock(flag, periods, seq_length, embed_dim, embed_dim_t, num_heads, ff_dim, num_layers)
            for _ in range(2)
        ])
        
        self.tof_blocks = nn.ModuleList([
            PeriodicBlock(flag, periods, seq_length, embed_dim, embed_dim_t, num_heads, ff_dim, num_layers)
            for _ in range(2)
        ])
        
        self.thm_blocks = nn.ModuleList([
            PeriodicBlock(flag, periods, seq_length, embed_dim, embed_dim_t, num_heads, ff_dim, num_layers)
            for _ in range(2)
        ])
        
        # Fusion layers
        self.sensor_fusion = nn.Linear(embed_dim * 3, embed_dim)  # Fuse 3 sensor types
        self.multi_sensor_fusion = nn.Linear(embed_dim * 6, embed_dim)  # Fuse multiple sensors of same type
        
        self.activation = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(seq_length * embed_dim, num_classes)

    def process_sensor_data(self, data, embedding_layer, processing_blocks):
        """Process single sensor type data through embedding and blocks"""
        B, n_sensors, L, n_vars = data.shape
        
        # Process each sensor separately
        sensor_features = []
        for i in range(n_sensors):
            sensor_data = data[:, i, :, :]  # [B, L, n_vars]
            
            # Embedding
            x = embedding_layer(sensor_data)  # [B, L, embed_dim]
            x = x.permute(0, 2, 1)  # [B, embed_dim, L]
            
            # Processing blocks
            for block in processing_blocks:
                x = self.layer_norm(block(x).permute(0, 2, 1)).permute(0, 2, 1)
            
            sensor_features.append(x)  # [B, embed_dim, L]
        
        # Fuse multiple sensors of the same type
        if n_sensors > 1:
            sensor_features = torch.stack(sensor_features, dim=1)  # [B, n_sensors, embed_dim, L]
            sensor_features = sensor_features.mean(dim=1)  # Average across sensors [B, embed_dim, L]
        else:
            sensor_features = sensor_features[0]
            
        return sensor_features

    def forward(self, imu_data, thm_data, tof_data):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data  
            thm_data: [B, 5, L, 1] - Thermal sensor data
        Returns:
            logits: [B, num_classes] - Classification logits
        """
        # Process each sensor type separately
        imu_features = self.process_sensor_data(imu_data, self.imu_embedding, self.imu_blocks)  # [B, embed_dim, L]
        tof_features = self.process_sensor_data(tof_data, self.tof_embedding, self.tof_blocks)  # [B, embed_dim, L]
        thm_features = self.process_sensor_data(thm_data, self.thm_embedding, self.thm_blocks)  # [B, embed_dim, L]
        
        # Sensor-level fusion
        fused_features = torch.cat([imu_features, tof_features, thm_features], dim=1)  # [B, 3*embed_dim, L]
        fused_features = fused_features.permute(0, 2, 1)  # [B, L, 3*embed_dim]
        fused_features = self.sensor_fusion(fused_features)  # [B, L, embed_dim]
        fused_features = fused_features.permute(0, 2, 1)  # [B, embed_dim, L]
        
        # Final processing
        x = self.activation(fused_features)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)  # [B, embed_dim * L]
        output = self.fc(x.float())
        
        return output
