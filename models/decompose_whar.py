import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.decompose import Embedding, DecomposeConvBlock
from modules.mamba import Mamba_Layer, Att_Layer
from modules.selfattn import FullAttention, AttentionLayer
from mamba_ssm import Mamba
from einops import rearrange

# https://arxiv.org/pdf/2501.10917v1 <- !!!

# B: batch size
# M: number of variables in the multivariate sequence
# L: length of the sequence (number of time steps)
# T: number of time steps after embedding, also can be considered as the number of patches after splitting
# D: number of channels per variable
# P: kernel size of the embedding layer
# S: stride of the embedding layer
class DecomposeWHAR(nn.Module):
    def __init__(self,
                 num_sensor,  # Number of sensors (N)
                 M,  # Number of variables in the multivariate sequence
                 L,  # Length of the input sequence (time steps)
                 D=64,  # Number of channels per variable
                 P=8,  # Kernel size of the embedding layer
                 S=4,  # Stride of the embedding layer
                 kernel_size=5,  # Kernel size for convolutional layers
                 r=1,  # A hyperparameter for decomposition (e.g., reduction ratio)
                 num_layers=2,  # Number of decomposition layers
                 num_m_layers=1,   # Number of mamba layers
                 num_a_layers=1):   # Number of attention layers
        super(DecomposeWHAR, self).__init__()

        self.num_layers = num_layers
        self.num_a_layers = num_a_layers
        self.num_m_layers = num_m_layers
        T = L // S  # Calculate the number of patches after embedding
        self.T = T
        self.D = D
        # Embedding layer to transform input sequences into higher dimensional representations
        self.embed_layer = Embedding(P, S, D)
        # Backbone consisting of multiple decomposition convolutional blocks
        self.backbone = nn.ModuleList([DecomposeConvBlock(M, D, kernel_size, r) for _ in range(num_layers)])
        # Fully connected output layer for classification
        # self.fc_out = nn.Linear(num_sensor * D * T, num_classes)
        self.dropout_prob = 0.6  # Dropout probability

        d_model = D * T  # Model dimension after embedding
        d_model_mamba = num_sensor * D  # Model dimension for Mamba preprocessing
        d_state = 16  # State dimension for Mamba
        d_conv = 4  # Convolutional dimension for Mamba
        dropout = 0.05  # Dropout for attention layers
        factor = 1  # Factor for the FullAttention layer
        n_heads = 8  # Number of attention heads
        self.d_layers = 1  # Number of attention layers

        # Mamba Block of Global Temporal Aggregation (GTA)
        self.mamba_preprocess = nn.ModuleList([
            Mamba_Layer(Mamba(d_model=d_model_mamba, d_state=d_state, d_conv=d_conv), d_model_mamba)
            for _ in range(num_m_layers)
        ])

        # Attention layers of Cross-Sensor Interaction (CSI)
        self.AM_layers = nn.ModuleList(
            [
                Att_Layer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=True),
                        d_model, n_heads),
                    d_model,
                    dropout
                )
                for i in range(self.d_layers)
            ]
        )
        # self.register_buffer(
        #     "centers", (torch.randn(num_classes, num_sensor*D * T).cuda())
        # )

    def forward(self, inputs):  # inputs: (B, N, L, M) - Batch size, Number of sensors, Sequence length, Number of variables
        B, N, L, M = inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3] # 64,5,24,9
        x = inputs.reshape(B*N, L, M)  # (B*N, L, M)
        x = x.permute(0, 2, 1)  # (B*N, M, L)

        x_emb = self.embed_layer(x)  # [B*N, M, L] -> [B*N, M, D, T]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B*N, M, D, L] -> [B*N, M, D, T]

        # Flatten
        x_emb = rearrange(x_emb, 'b m d t -> b m (d t)', b=B*N, m=M)  # [B*N, M, D, T] -> [B*N, M, D*T]

        # Aggregate over the sensor dimension and apply classification head
        x_emb = x_emb.mean(dim=1)  # [B*N, M, D*T] -> [B*N, D*T] 64*5,D*T
        x_emb = x_emb.reshape(inputs.shape[0],inputs.shape[1],-1) # B,N,D*T (64,5,D*T)

        # Reshape to B,T,N*D (64, T, 5*D)
        x_emb = x_emb.reshape(inputs.shape[0],inputs.shape[1],self.D,self.T) # B,N,D,T (64,5,D,T)
        x_emb = x_emb.permute(0,3,1,2)
        x_emb = x_emb.reshape(inputs.shape[0],x_emb.shape[1],-1) # B,T,N*D (64,T,5*D)

        # Mamba Input
        for i in range(0,self.num_m_layers):
            x_emb = self.mamba_preprocess[i](x_emb)  # B,T,N*D

        # Reshape to B, N, D*T
        x_emb = x_emb.reshape(inputs.shape[0], x_emb.shape[1], inputs.shape[1], self.D)  # B,T,N,D
        x_emb = x_emb.permute(0, 2, 3, 1)  # B,N,D,T
        x_emb = x_emb.reshape(inputs.shape[0], x_emb.shape[1], -1)  # B,N,D*T

        # Attention
        for i in range(self.num_a_layers):
            x_emb, output_attention = self.AM_layers[i](x_emb, None)  # B,N,D*T

        x = x_emb

        x = x.reshape(inputs.shape[0],-1)

        # x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # pred = self.fc_out(x)  # [B, D*T] -> [B, num_classes]

        return x #, pred # output: [B, num_classes]

# just DecomposeWHAR for each sensor lol
class MultiSensorDecomposeWHAR(nn.Module):
    def __init__(self, 
                 imu_num_sensor=1, imu_M=7, imu_L=110, imu_D=32,
                 thm_num_sensor=5, thm_M=1, thm_L=110, thm_D=32,
                 tof_num_sensor=5, tof_M=64, tof_L=110, tof_D=64,
                 num_classes=18, S=4, use_cross_sensor=True):
        super().__init__()
        
        self.S = S
        self.use_cross_sensor = use_cross_sensor
        
        self.imu_dwhar = DecomposeWHAR(
            num_sensor=imu_num_sensor, M=imu_M, L=imu_L, D=imu_D, S=S
        )
        self.thm_dwhar = DecomposeWHAR(
            num_sensor=thm_num_sensor, M=thm_M, L=thm_L, D=thm_D, S=S
        )
        self.tof_dwhar = DecomposeWHAR(
            num_sensor=tof_num_sensor, M=tof_M, L=tof_L, D=tof_D, S=S
        )
        
        imu_size = imu_num_sensor * imu_D * (imu_L // S)
        thm_size = thm_num_sensor * thm_D * (thm_L // S)
        tof_size = tof_num_sensor * tof_D * (tof_L // S)
        
        if use_cross_sensor:
            common_dim = 512
            self.imu_proj = nn.Linear(imu_size, common_dim)
            self.thm_proj = nn.Linear(thm_size, common_dim)
            self.tof_proj = nn.Linear(tof_size, common_dim)
            
            self.cross_sensor_attention = nn.MultiheadAttention(
                embed_dim=common_dim, num_heads=8, batch_first=True
            )
            
            self.final_proj = nn.Sequential(
                nn.Linear(3 * common_dim, common_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(common_dim, num_classes)
            )
        else:
            total_size = imu_size + thm_size + tof_size
            self.classifier = nn.Linear(total_size, num_classes)
    
    def forward(self, imu_data, thm_data, tof_data):
        imu_x = self.imu_dwhar(imu_data)  # (B, imu_size)
        thm_x = self.thm_dwhar(thm_data)  # (B, thm_size)
        tof_x = self.tof_dwhar(tof_data)  # (B, tof_size)
        
        if self.use_cross_sensor:
            imu_proj = self.imu_proj(imu_x)  # (B, common_dim)
            thm_proj = self.thm_proj(thm_x)  # (B, common_dim)
            tof_proj = self.tof_proj(tof_x)  # (B, common_dim)
            
            sensor_sequence = torch.stack([imu_proj, thm_proj, tof_proj], dim=1)  # (B, 3, common_dim)
            
            attended_features, attention_weights = self.cross_sensor_attention(
                sensor_sequence, sensor_sequence, sensor_sequence
            )  # (B, 3, common_dim)
            
            combined = attended_features.flatten(1)  # (B, 3*common_dim)
            pred = self.final_proj(combined)
            
            return pred
        else:
            comb_x = torch.cat([imu_x, thm_x, tof_x], dim=1)
            pred = self.classifier(comb_x)
            return pred

class IMU_DecomposeWHAR(nn.Module):
    def __init__(
            self, 
            num_sensor=1,
            M=7,
            L=110,
            num_classes=18,
            D=64,
            S=4,
        ):
        super().__init__()

        self.model = DecomposeWHAR(
            num_sensor=num_sensor,
            M=M,
            L=L,
            D=D,
            S=S
        )

        self.classifier = nn.Linear(num_sensor * D * (L // S), num_classes)
    
    def forward(self, imu_data):
        x = self.model(imu_data)
        pred = self.classifier(x)
        return pred