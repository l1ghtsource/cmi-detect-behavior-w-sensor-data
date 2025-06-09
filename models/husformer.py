import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder

class Original_HUSFORMERModel(nn.Module):
    def __init__(self, hyp_params):
        super(Original_HUSFORMERModel, self).__init__()
        self.orig_d_m1, self.orig_d_m2, self.orig_d_m3  = hyp_params.orig_d_m1, hyp_params.orig_d_m2, hyp_params.orig_d_m3
        self.d_m = 30
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        combined_dim = 30     
        output_dim = hyp_params.output_dim
        self.channels = hyp_params.m1_len+hyp_params.m2_len+hyp_params.m3_len
        
        # 1. Temporal convolutional layers
        self.proj_m1 = nn.Conv1d(self.orig_d_m1, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m2 = nn.Conv1d(self.orig_d_m2, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m3 = nn.Conv1d(self.orig_d_m3, self.d_m, kernel_size=1, padding=0, bias=False)
        self.final_conv = nn.Conv1d(self.channels, 1, kernel_size=1, padding=0, bias=False)
        
        # 2. Cross-modal Attentions
        self.trans_m1_all = self.get_network(self_type='m1_all', layers=3)
        self.trans_m2_all = self.get_network(self_type='m2_all', layers=3)
        self.trans_m3_all = self.get_network(self_type='m3_all', layers=3)
        
        # 3. Self Attentions
        self.trans_final = self.get_network(self_type='policy', layers=5)
        
        # 4. Projection layers
        self.proj1 = self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['m1_all','m2_all','m3_all','policy']:
            embed_dim, attn_dropout = self.d_m, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self,m1,m2,m3):

        m_1 = m1.transpose(1, 2)
        m_2 = m2.transpose(1, 2)
        m_3 = m3.transpose(1, 2)

        proj_x_m1 = m_1 if self.orig_d_m1 == self.d_m else self.proj_m1(m_1)
        proj_x_m2 = m_2 if self.orig_d_m2 == self.d_m else self.proj_m2(m_2)
        proj_x_m3 = m_3 if self.orig_d_m3 == self.d_m else self.proj_m3(m_3)

        proj_x_m1 = proj_x_m1.permute(2, 0, 1)
        proj_x_m2 = proj_x_m2.permute(2, 0, 1)
        proj_x_m3 = proj_x_m3.permute(2, 0, 1)
        
        proj_all = torch.cat([proj_x_m1 , proj_x_m2 , proj_x_m3], dim=0)
            
        m1_with_all = self.trans_m1_all(proj_x_m1, proj_all, proj_all)  
        m2_with_all = self.trans_m2_all(proj_x_m2, proj_all, proj_all)  
        m3_with_all = self.trans_m3_all(proj_x_m3, proj_all, proj_all)  
 
        last_hs1 = torch.cat([m1_with_all, m2_with_all, m3_with_all] , dim = 0)
        last_hs2 = self.trans_final(last_hs1).permute(1, 0, 2)
        last_hs = self.final_conv(last_hs2).squeeze(1)

        output = self.out_layer(last_hs)

        return output, last_hs
    
class MultiSensor_HUSFORMER_v1(nn.Module):
    def __init__(self, 
                 num_heads=8,
                 layers=4,
                 attn_dropout=0.1,
                 relu_dropout=0.1,
                 res_dropout=0.1,
                 out_dropout=0.1,
                 embed_dropout=0.1,
                 attn_mask=False,
                 output_dim=2,
                 d_m=30):
        super(MultiSensor_HUSFORMER_v1, self).__init__()
        
        # Store all parameters
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask
        self.output_dim = output_dim
        self.d_m = d_m
        
        # Original dimensions for each sensor type
        self.orig_d_imu = 7  # IMU variables
        self.orig_d_tof = 64 * 5  # ToF: 5 sensors * 64 variables each = 320
        self.orig_d_thm = 1 * 5   # Thermal: 5 sensors * 1 variable each = 5
        
        # Combined dimension and channels
        self.combined_dim = d_m     
        self.channels = 3  # IMU + ToF + Thermal modalities
        
        # 1. Temporal convolutional layers for each modality
        self.proj_imu = nn.Conv1d(self.orig_d_imu, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_tof = nn.Conv1d(self.orig_d_tof, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_thm = nn.Conv1d(self.orig_d_thm, self.d_m, kernel_size=1, padding=0, bias=False)
        
        self.final_conv = nn.Conv1d(self.channels, 1, kernel_size=1, padding=0, bias=False)
        
        # 2. Cross-modal Attentions
        self.trans_imu_all = self.get_network(self_type='imu_all', layers=3)
        self.trans_tof_all = self.get_network(self_type='tof_all', layers=3)
        self.trans_thm_all = self.get_network(self_type='thm_all', layers=3)
        
        # 3. Self Attentions
        self.trans_final = self.get_network(self_type='policy', layers=5)
        
        # 4. Projection layers
        self.proj1 = self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, self.output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['imu_all', 'tof_all', 'thm_all', 'policy']:
            embed_dim, attn_dropout = self.d_m, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    def process_sensor_data(self, sensor_data, num_sensors):
        """
        Process multi-sensor data by reshaping and concatenating
        Args:
            sensor_data: [B, num_sensors, L, features]
            num_sensors: number of sensors
        Returns:
            processed_data: [B, num_sensors * features, L]
        """
        B, S, L, F = sensor_data.shape
        # Reshape to [B, S*F, L] to concatenate all sensor features
        return sensor_data.reshape(B, S * F, L)
            
    def forward(self, imu_data, thm_data, tof_data):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data  
            thm_data: [B, 5, L, 1] - Thermal sensor data
        """
        B, _, L, _ = imu_data.shape
        
        # Process each sensor type
        # IMU: [B, 1, L, 7] -> [B, 7, L]
        imu_processed = imu_data.squeeze(1).transpose(1, 2)  # [B, 7, L]
        
        # ToF: [B, 5, L, 64] -> [B, 320, L] (5*64=320)
        tof_processed = self.process_sensor_data(tof_data, 5)  # [B, 320, L]
        
        # Thermal: [B, 5, L, 1] -> [B, 5, L]
        thm_processed = self.process_sensor_data(thm_data, 5)  # [B, 5, L]
        
        # Apply projections to get same embedding dimension
        proj_x_imu = self.proj_imu(imu_processed)  # [B, 30, L]
        proj_x_tof = self.proj_tof(tof_processed)  # [B, 30, L]
        proj_x_thm = self.proj_thm(thm_processed)  # [B, 30, L]
        
        # Transpose for transformer: [L, B, 30]
        proj_x_imu = proj_x_imu.permute(2, 0, 1)  # [L, B, 30]
        proj_x_tof = proj_x_tof.permute(2, 0, 1)  # [L, B, 30]
        proj_x_thm = proj_x_thm.permute(2, 0, 1)  # [L, B, 30]
        
        # Concatenate all modalities: [3*L, B, 30]
        proj_all = torch.cat([proj_x_imu, proj_x_tof, proj_x_thm], dim=0)
        
        # Cross-modal attention: each modality attends to all modalities
        imu_with_all = self.trans_imu_all(proj_x_imu, proj_all, proj_all)  # [L, B, 30]
        tof_with_all = self.trans_tof_all(proj_x_tof, proj_all, proj_all)  # [L, B, 30]
        thm_with_all = self.trans_thm_all(proj_x_thm, proj_all, proj_all)  # [L, B, 30]
        
        # Concatenate attended features: [3*L, B, 30]
        last_hs1 = torch.cat([imu_with_all, tof_with_all, thm_with_all], dim=0)
        
        # Final self-attention and reshape: [B, 3*L, 30]
        last_hs2 = self.trans_final(last_hs1).permute(1, 0, 2)
        
        # Reduce sequence dimension: [B, 3, 30] -> [B, 1, 30] -> [B, 30]
        last_hs2_reshaped = last_hs2.view(B, 3, L, 30).mean(dim=2)  # Average over time
        last_hs = self.final_conv(last_hs2_reshaped.transpose(1, 2)).squeeze(1)  # [B, 30]
        
        # Final classification
        output = self.out_layer(last_hs)
        
        return output
    
class SingleSensor_HUSFORMER_v1(nn.Module):
    def __init__(self, 
                 num_heads=8,
                 layers=4,
                 attn_dropout=0.1,
                 relu_dropout=0.1,
                 res_dropout=0.1,
                 out_dropout=0.1,
                 embed_dropout=0.1,
                 attn_mask=False,
                 output_dim=2,
                 d_m=30):
        super(SingleSensor_HUSFORMER_v1, self).__init__()
        
        # Store all parameters
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask
        self.output_dim = output_dim
        self.d_m = d_m
        
        # Original dimension for IMU sensor only
        self.orig_d_imu = 7  # IMU variables
        
        # Combined dimension and channels
        self.combined_dim = d_m     
        self.channels = 1  # Only IMU modality
        
        # 1. Temporal convolutional layer for IMU modality
        self.proj_imu = nn.Conv1d(self.orig_d_imu, self.d_m, kernel_size=1, padding=0, bias=False)
        
        self.final_conv = nn.Conv1d(self.channels, 1, kernel_size=1, padding=0, bias=False)
        
        # 2. Self Attention (no cross-modal needed for single sensor)
        self.trans_imu_all = self.get_network(self_type='imu_all', layers=3)
        
        # 3. Final Self Attention
        self.trans_final = self.get_network(self_type='policy', layers=5)
        
        # 4. Projection layers
        self.proj1 = self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, self.output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['imu_all', 'policy']:
            embed_dim, attn_dropout = self.d_m, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, imu_data):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
        """
        B, _, L, _ = imu_data.shape
        
        # Process IMU data
        # IMU: [B, 1, L, 7] -> [B, 7, L]
        imu_processed = imu_data.squeeze(1).transpose(1, 2)  # [B, 7, L]
        
        # Apply projection to get embedding dimension
        proj_x_imu = self.proj_imu(imu_processed)  # [B, 30, L]
        
        # Transpose for transformer: [L, B, 30]
        proj_x_imu = proj_x_imu.permute(2, 0, 1)  # [L, B, 30]
        
        # Since we only have one modality, proj_all is just proj_x_imu
        proj_all = proj_x_imu  # [L, B, 30]
        
        # Self-attention: IMU attends to itself (no cross-modal)
        imu_with_all = self.trans_imu_all(proj_x_imu, proj_all, proj_all)  # [L, B, 30]
        
        # Since we only have one modality, last_hs1 is just imu_with_all
        last_hs1 = imu_with_all  # [L, B, 30]
        
        # Final self-attention and reshape: [B, L, 30]
        last_hs2 = self.trans_final(last_hs1).permute(1, 0, 2)
        
        # Reduce sequence dimension: [B, 1, 30] -> [B, 30]
        last_hs2_reshaped = last_hs2.mean(dim=1)  # Average over time: [B, 30]
        last_hs2_reshaped = last_hs2_reshaped.unsqueeze(1)  # [B, 1, 30]
        last_hs = self.final_conv(last_hs2_reshaped.transpose(1, 2)).squeeze(1)  # [B, 30]
        
        # Final classification
        output = self.out_layer(last_hs)
        
        return output