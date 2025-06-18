import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder

# orig -> https://arxiv.org/pdf/2209.15182 !!!

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

# imu + tof + thm modification 
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
                 output_dim=18,
                 d_m=64):
        super(MultiSensor_HUSFORMER_v1, self).__init__()

        self.d_m = d_m
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask

        self.channels = 1 + 5 + 5  # IMU(1) + ToF(5) + THM(5)

        # Projection layers for modalities
        self.proj_imu = nn.Conv1d(7, d_m, kernel_size=1, padding=0, bias=False)
        self.proj_tof = nn.Conv1d(64, d_m, kernel_size=1, padding=0, bias=False)
        self.proj_thm = nn.Conv1d(1, d_m, kernel_size=1, padding=0, bias=False)

        self.final_conv = nn.Conv1d(self.channels, 1, kernel_size=1, padding=0, bias=False)

        self.trans_imu_all = self.get_network('imu_all', layers=3)
        self.trans_tof_all = self.get_network('tof_all', layers=3)
        self.trans_thm_all = self.get_network('thm_all', layers=3)

        self.trans_final = self.get_network('policy', layers=5)

        self.proj1 = self.proj2 = nn.Linear(d_m, d_m)
        self.out_layer = nn.Linear(d_m, output_dim)
        self.out_layer2 = nn.Linear(d_m, 2)

    def get_network(self, self_type, layers):
        return TransformerEncoder(
            embed_dim=self.d_m,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def forward(self, imu_data, thm_data, tof_data, pad_mask=None):
        # Shapes: [B, C, L, F]
        B, _, L, _ = imu_data.shape

        imu = imu_data[:, 0]       # [B, L, 7]
        tof = tof_data.reshape(B * 5, L, 64)
        thm = thm_data.reshape(B * 5, L, 1)

        imu = self.proj_imu(imu.transpose(1, 2))       # [B, d_m, L]
        tof = self.proj_tof(tof.transpose(1, 2))       # [B*5, d_m, L]
        thm = self.proj_thm(thm.transpose(1, 2))       # [B*5, d_m, L]

        imu = imu.permute(2, 0, 1)                     # [L, B, d_m]
        tof = tof.view(L, B, 5, self.d_m).permute(0, 2, 1, 3).reshape(L, B * 5, self.d_m)
        thm = thm.view(L, B, 5, self.d_m).permute(0, 2, 1, 3).reshape(L, B * 5, self.d_m)

        all_modal = torch.cat([imu, tof, thm], dim=1)

        imu_out = self.trans_imu_all(imu, all_modal, all_modal)
        tof_out = self.trans_tof_all(tof, all_modal, all_modal)
        thm_out = self.trans_thm_all(thm, all_modal, all_modal)

        merged = torch.cat([imu_out, tof_out, thm_out], dim=1)
        merged = self.trans_final(merged).permute(1, 0, 2)  # [B_all, L, d_m]
        final_out = self.final_conv(merged).squeeze(1)      # [B, d_m]
        output = self.out_layer(final_out)
        output2 = self.out_layer2(final_out)

        return output, output2
    
# imu only modification 
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
                 output_dim=18,
                 d_m=256):
        super(SingleSensor_HUSFORMER_v1, self).__init__()

        self.d_m = d_m
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask

        self.proj_imu = nn.Conv1d(7, d_m, kernel_size=1, padding=0, bias=False)
        self.trans = self.get_network('imu_only', layers)
        self.out_layer = nn.Linear(d_m, output_dim)
        self.out_layer2 = nn.Linear(d_m, 2)

    def get_network(self, self_type, layers):
        return TransformerEncoder(
            embed_dim=self.d_m,
            num_heads=self.num_heads,
            layers=layers,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def forward(self, imu_data, pad_mask=None):
        # [B, 1, L, 7] -> [B, L, 7]
        imu = imu_data[:, 0]
        imu = self.proj_imu(imu.transpose(1, 2))       # [B, d_m, L]
        imu = imu.permute(2, 0, 1)                     # [L, B, d_m]
        out = self.trans(imu).permute(1, 0, 2)         # [B, L, d_m]
        out = out.mean(dim=1)                          # simple pooling
        output = self.out_layer(out)
        output2 = self.out_layer2(out)
        return output, output2

# imu + tof + thm modification (tof as 2d)
class MultiSensor_HUSFORMER_v2(nn.Module):
    def __init__(self, 
                 num_heads=8,
                 layers=4,
                 attn_dropout=0.1,
                 relu_dropout=0.1,
                 res_dropout=0.1,
                 out_dropout=0.1,
                 embed_dropout=0.1,
                 attn_mask=False,
                 output_dim=18,
                 d_m=64):
        super(MultiSensor_HUSFORMER_v2, self).__init__()

        self.d_m = d_m
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask

        self.orig_d_imu = 7  # IMU variables
        self.orig_d_tof = 1  # ToF: 8x8 maps, 1 channel input
        self.orig_d_thm = 1 * 5   # Thermal: 5 sensors * 1 variable each = 5
        
        # Combined dimension and channels
        self.combined_dim = d_m     
        self.channels = 3  # IMU + ToF + Thermal modalities

        self.channels = 1 + 5 + 5  # IMU(1) + ToF(5) + THM(5)

        # Projection layers for modalities
        self.proj_imu = nn.Conv1d(7, d_m, kernel_size=1, padding=0, bias=False)
        self.proj_tof = nn.Conv2d(1, d_m, kernel_size=(3, 3), padding=1, bias=False)
        self.proj_thm = nn.Conv1d(1, d_m, kernel_size=1, padding=0, bias=False)

        self.final_conv = nn.Conv1d(self.channels, 1, kernel_size=1, padding=0, bias=False)

        self.trans_imu_all = self.get_network('imu_all', layers=3)
        self.trans_tof_all = self.get_network('tof_all', layers=3)
        self.trans_thm_all = self.get_network('thm_all', layers=3)

        self.trans_final = self.get_network('policy', layers=5)

        self.proj1 = self.proj2 = nn.Linear(d_m, d_m)
        self.out_layer = nn.Linear(d_m, output_dim)
        self.out_layer2 = nn.Linear(d_m, 2)

    def get_network(self, self_type, layers):
        return TransformerEncoder(
            embed_dim=self.d_m,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def forward(self, imu_data, thm_data, tof_data, pad_mask=None):
        B, _, L, _ = imu_data.shape

        imu = imu_data[:, 0]                     # [B, L, 7]
        thm = thm_data.reshape(B * 5, L, 1)      # [B*5, L, 1]

        # ToF: [B, 5, L, 64] → [B*5*L, 8, 8]
        tof = tof_data.reshape(B * 5 * L, 8, 8).unsqueeze(1)  # [B*5*L, 1, 8, 8]
        tof = self.proj_tof(tof)                             # [B*5*L, d_m, 8, 8]
        tof = torch.mean(tof, dim=[2, 3])                    # Global avg pooling → [B*5*L, d_m]
        tof = tof.view(B * 5, L, self.d_m).transpose(1, 2)   # [B*5, d_m, L]

        imu = self.proj_imu(imu.transpose(1, 2))             # [B, d_m, L]
        thm = self.proj_thm(thm.transpose(1, 2))             # [B*5, d_m, L]

        imu = imu.permute(2, 0, 1)                           # [L, B, d_m]
        tof = tof.permute(2, 0, 1)                           # [L, B*5, d_m]
        thm = thm.permute(2, 0, 1)                           # [L, B*5, d_m]

        all_modal = torch.cat([imu, tof, thm], dim=1)

        imu_out = self.trans_imu_all(imu, all_modal, all_modal)
        tof_out = self.trans_tof_all(tof, all_modal, all_modal)
        thm_out = self.trans_thm_all(thm, all_modal, all_modal)

        merged = torch.cat([imu_out, tof_out, thm_out], dim=1)
        merged = self.trans_final(merged).permute(1, 0, 2)   # [B_all, L, d_m]

        final_out = self.final_conv(merged).squeeze(1)       # [B, d_m]
        output = self.out_layer(final_out)
        output2 = self.out_layer2(final_out)

        return output, output2