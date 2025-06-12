import torch
import torch.nn as nn
import torch.nn.functional as F

class Wave_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                         padding=int((dilation_rate*(kernel_size-1))/2), 
                         dilation=dilation_rate, padding_mode='replicate'))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                         padding=int((dilation_rate*(kernel_size-1))/2), 
                         dilation=dilation_rate, padding_mode='replicate'))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(SEModule, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        s = F.adaptive_avg_pool1d(x, 1)
        s = self.conv(s)
        x *= torch.sigmoid(s)
        return x

class WaveNet_SingleSensor_v1(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        dropout_rate = 0.1
        inch = 7  # imu

        self.conv1d_1 = nn.Conv1d(inch, 32, kernel_size=1, stride=1, dilation=1, 
                                 padding=0, padding_mode='replicate')
        self.batch_norm_conv_1 = nn.BatchNorm1d(32)
        self.dropout_conv_1 = nn.Dropout(dropout_rate)

        self.conv1d_2 = nn.Conv1d(inch+16+32+64+128, 32, kernel_size=1, stride=1, 
                                 dilation=1, padding=0, padding_mode='replicate')
        self.batch_norm_conv_2 = nn.BatchNorm1d(32)
        self.dropout_conv_2 = nn.Dropout(dropout_rate)

        self.wave_block1 = Wave_Block(32, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(inch+16, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(inch+16+32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(inch+16+32+64, 128, 1, kernel_size)

        self.se_module1 = SEModule(16)
        self.se_module2 = SEModule(32)
        self.se_module3 = SEModule(64)
        self.se_module4 = SEModule(128)        

        self.batch_norm_1 = nn.BatchNorm1d(16)
        self.batch_norm_2 = nn.BatchNorm1d(32)
        self.batch_norm_3 = nn.BatchNorm1d(64)
        self.batch_norm_4 = nn.BatchNorm1d(128)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dropout_3 = nn.Dropout(dropout_rate)
        self.dropout_4 = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(inch+16+32+64, 32, 1, batch_first=True, bidirectional=True)
        self.fc0 = nn.Linear(64, 18)
        self.fc1 = nn.Linear(64, 4)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, imu_data, pad_mask=None):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            pad_mask: [B, L] - padding mask (1=valid, 0=padding) [optional!]
        """
        x = imu_data.squeeze(1).permute(0, 2, 1)  # [B, 1, L, 7] -> [B, L, 7] -> [B, 7, L]
        
        if pad_mask is not None:
            mask_expanded = pad_mask.unsqueeze(1).expand(-1, x.size(1), -1).float()
            x = x * mask_expanded

        x0 = self.conv1d_1(x)
        x0 = F.relu(x0)
        x0 = self.batch_norm_conv_1(x0)
        x0 = self.dropout_conv_1(x0)

        x1 = self.wave_block1(x0)
        x1 = self.batch_norm_1(x1)
        x1 = self.dropout_1(x1)
        x1 = self.se_module1(x1)
        x2_base = torch.cat([x1, x], dim=1)

        x2 = self.wave_block2(x2_base)
        x2 = self.batch_norm_2(x2)
        x2 = self.dropout_2(x2)
        x2 = self.se_module2(x2)
        x3_base = torch.cat([x2_base, x2], dim=1)

        x3 = self.wave_block3(x3_base)
        x3 = self.batch_norm_3(x3)
        x3 = self.dropout_3(x3)
        x3 = self.se_module3(x3)
        x4_base = torch.cat([x3_base, x3], dim=1)

        x4 = self.wave_block4(x4_base)
        x4 = self.batch_norm_4(x4)
        x4 = self.dropout_4(x4)
        x4 = self.se_module4(x4)
        x5_base = torch.cat([x4_base, x4], dim=1)

        x5 = self.conv1d_2(x5_base)
        x5 = F.relu(x5)
        x5 = self.batch_norm_conv_2(x5)
        x5 = self.dropout_conv_2(x5)

        lstm_input = x4_base.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
        
        if pad_mask is not None:
            lengths = pad_mask.sum(dim=1).cpu()
            lstm_input = nn.utils.rnn.pack_padded_sequence(
                lstm_input, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(lstm_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(lstm_input)
        
        out0 = self.fc0(lstm_out)  # [B, L, 18]
        out2 = self.fc2(lstm_out)  # [B, L, 2]

        if pad_mask is not None:
            mask_output = pad_mask.unsqueeze(-1).float()
            out0_masked = out0 * mask_output.expand_as(out0)
            out2_masked = out2 * mask_output.expand_as(out2)
            
            valid_lengths = pad_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
            out0 = out0_masked.sum(dim=1) / valid_lengths  # [B, 18]
            out2 = out2_masked.sum(dim=1) / valid_lengths  # [B, 2]
        else:
            out0 = out0.mean(dim=1)  # [B, 18]
            out2 = out2.mean(dim=1)  # [B, 2]

        return out0, out2
    
class IntraSensorModule(nn.Module):
    def __init__(self, input_features, output_features, num_sensors, kernel_size=3):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_features = input_features
        self.output_features = output_features
        
        self.sensor_processors = nn.ModuleList([
            Wave_Block(input_features, output_features // 2, 4, kernel_size)
            for _ in range(num_sensors)
        ])
        
        self.inter_sensor_fusion = Wave_Block(
            num_sensors * (output_features // 2), 
            output_features, 
            2, 
            kernel_size
        )
        
        self.batch_norm = nn.BatchNorm1d(output_features)
        self.dropout = nn.Dropout(0.1)
        self.se_module = SEModule(output_features)

    def forward(self, x):
        """
        Args:
            x: [B, num_sensors, L, input_features]
        Returns:
            [B, output_features, L]
        """
        B, num_sensors, L, features = x.shape
        
        sensor_outputs = []
        for i in range(num_sensors):
            # [B, L, features] -> [B, features, L]
            sensor_data = x[:, i, :, :].permute(0, 2, 1)
            sensor_out = self.sensor_processors[i](sensor_data)
            sensor_outputs.append(sensor_out)
        
        combined = torch.cat(sensor_outputs, dim=1)  # [B, num_sensors * (output_features // 2), L]
        
        fused = self.inter_sensor_fusion(combined)
        fused = self.batch_norm(fused)
        fused = self.dropout(fused)
        fused = self.se_module(fused)
        
        return fused

class InterSensorFusionModule(nn.Module):
    def __init__(self, imu_features, tof_features, thm_features, output_features, kernel_size=3):
        super().__init__()
        
        total_features = imu_features + tof_features + thm_features
        
        self.fusion_wave_block1 = Wave_Block(total_features, output_features, 8, kernel_size)
        self.fusion_wave_block2 = Wave_Block(output_features, output_features, 4, kernel_size)
        
        self.batch_norm1 = nn.BatchNorm1d(output_features)
        self.batch_norm2 = nn.BatchNorm1d(output_features)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
        self.se_module1 = SEModule(output_features)
        self.se_module2 = SEModule(output_features)

    def forward(self, imu_features, tof_features, thm_features):
        """
        Args:
            imu_features: [B, imu_features, L]
            tof_features: [B, tof_features, L]  
            thm_features: [B, thm_features, L]
        Returns:
            [B, output_features, L]
        """
        combined = torch.cat([imu_features, tof_features, thm_features], dim=1)
        
        x = self.fusion_wave_block1(combined)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.se_module1(x)
        
        x = self.fusion_wave_block2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.se_module2(x)
        
        return x

class WaveNet_MultiSensor_v1(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        dropout_rate = 0.1
        
        imu_features = 7
        self.imu_conv1d_1 = nn.Conv1d(imu_features, 32, kernel_size=1, stride=1, 
                                     dilation=1, padding=0, padding_mode='replicate')
        self.imu_batch_norm_conv_1 = nn.BatchNorm1d(32)
        self.imu_dropout_conv_1 = nn.Dropout(dropout_rate)
        
        self.imu_wave_block1 = Wave_Block(32, 16, 8, kernel_size)
        self.imu_wave_block2 = Wave_Block(imu_features + 16, 32, 4, kernel_size)
        self.imu_wave_block3 = Wave_Block(imu_features + 16 + 32, 64, 2, kernel_size)
        
        self.imu_batch_norm_1 = nn.BatchNorm1d(16)
        self.imu_batch_norm_2 = nn.BatchNorm1d(32)
        self.imu_batch_norm_3 = nn.BatchNorm1d(64)
        
        self.imu_dropout_1 = nn.Dropout(dropout_rate)
        self.imu_dropout_2 = nn.Dropout(dropout_rate)
        self.imu_dropout_3 = nn.Dropout(dropout_rate)
        
        self.imu_se_module1 = SEModule(16)
        self.imu_se_module2 = SEModule(32)
        self.imu_se_module3 = SEModule(64)
        
        self.tof_intra_module = IntraSensorModule(
            input_features=64, 
            output_features=128, 
            num_sensors=5, 
            kernel_size=kernel_size
        )
        
        self.thm_intra_module = IntraSensorModule(
            input_features=1, 
            output_features=64, 
            num_sensors=5, 
            kernel_size=kernel_size
        )
        
        self.inter_sensor_fusion = InterSensorFusionModule(
            imu_features=64, 
            tof_features=128, 
            thm_features=64,  
            output_features=256,
            kernel_size=kernel_size
        )
        
        self.lstm = nn.LSTM(256, 64, 1, batch_first=True, bidirectional=True)
        self.fc0 = nn.Linear(128, 18)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, imu_data, tof_data, thm_data, pad_mask=None):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data
            thm_data: [B, 5, L, 1] - Thermal sensor data
            pad_mask: [B, L] - padding mask (1=valid, 0=padding) [optional!]
        """
        imu_x = imu_data.squeeze(1).permute(0, 2, 1)  # [B, 1, L, 7] -> [B, 7, L]
        
        if pad_mask is not None:
            mask_expanded = pad_mask.unsqueeze(1).expand(-1, imu_x.size(1), -1).float()
            imu_x = imu_x * mask_expanded

        imu_x0 = self.imu_conv1d_1(imu_x)
        imu_x0 = F.relu(imu_x0)
        imu_x0 = self.imu_batch_norm_conv_1(imu_x0)
        imu_x0 = self.imu_dropout_conv_1(imu_x0)

        imu_x1 = self.imu_wave_block1(imu_x0)
        imu_x1 = self.imu_batch_norm_1(imu_x1)
        imu_x1 = self.imu_dropout_1(imu_x1)
        imu_x1 = self.imu_se_module1(imu_x1)
        imu_x2_base = torch.cat([imu_x1, imu_x], dim=1)

        imu_x2 = self.imu_wave_block2(imu_x2_base)
        imu_x2 = self.imu_batch_norm_2(imu_x2)
        imu_x2 = self.imu_dropout_2(imu_x2)
        imu_x2 = self.imu_se_module2(imu_x2)
        imu_x3_base = torch.cat([imu_x2_base, imu_x2], dim=1)

        imu_x3 = self.imu_wave_block3(imu_x3_base)
        imu_x3 = self.imu_batch_norm_3(imu_x3)
        imu_x3 = self.imu_dropout_3(imu_x3)
        imu_features = self.imu_se_module3(imu_x3)  # [B, 64, L]
        
        tof_features = self.tof_intra_module(tof_data)  # [B, 128, L]
        thm_features = self.thm_intra_module(thm_data)  # [B, 64, L]
        
        fused_features = self.inter_sensor_fusion(
            imu_features, tof_features, thm_features
        )  # [B, 256, L]
        
        lstm_input = fused_features.permute(0, 2, 1)  # [B, L, 256]
        
        if pad_mask is not None:
            lengths = pad_mask.sum(dim=1).cpu()
            lstm_input = nn.utils.rnn.pack_padded_sequence(
                lstm_input, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(lstm_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(lstm_input)
        
        out0 = self.fc0(lstm_out)  # [B, L, 18]
        out2 = self.fc2(lstm_out)  # [B, L, 2]

        if pad_mask is not None:
            mask_output = pad_mask.unsqueeze(-1).float()
            out0_masked = out0 * mask_output.expand_as(out0)
            out2_masked = out2 * mask_output.expand_as(out2)
            
            valid_lengths = pad_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
            out0 = out0_masked.sum(dim=1) / valid_lengths  # [B, 18]
            out2 = out2_masked.sum(dim=1) / valid_lengths  # [B, 2]
        else:
            out0 = out0.mean(dim=1)  # [B, 18]
            out2 = out2.mean(dim=1)  # [B, 2]

        return out0, out2