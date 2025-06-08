import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.decompose import Embedding, ToFEmbedding, DecomposeConvBlock
from modules.mamba import Mamba_Layer, Att_Layer
from modules.selfattn import FullAttention, AttentionLayer
from mamba_ssm import Mamba
from einops import rearrange
from configs.config import cfg

# https://arxiv.org/pdf/2501.10917v1 <- !!!

# B: batch size
# M: number of variables in the multivariate sequence
# L: length of the sequence (number of time steps)
# T: number of time steps after embedding, also can be considered as the number of patches after splitting
# D: number of channels per variable
# P: kernel size of the embedding layer
# S: stride of the embedding layer
class DecomposeWHAR_Extractor(nn.Module):
    def __init__(self,
                 num_sensor,  # Number of sensors (N)
                 M,  # Number of variables in the multivariate sequence
                 L,  # Length of the input sequence (time steps)
                 D=cfg.ddim,  # Number of channels per variable
                 P=cfg.emb_kernel_size,  # Kernel size of the embedding layer
                 S=cfg.stride,  # Stride of the embedding layer
                 kernel_size=cfg.kernel_size,  # Kernel size for convolutional layers
                 r=cfg.reduction_ratio,  # A hyperparameter for decomposition (e.g., reduction ratio)
                 num_layers=cfg.num_layers,  # Number of decomposition layers
                 num_m_layers=cfg.num_m_layers,   # Number of mamba layers
                 num_a_layers=cfg.num_a_layers):   # Number of attention layers
        super(DecomposeWHAR_Extractor, self).__init__()

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
        B, N, L, M = inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3] # bs,n_sensors,seq_len,n_vars
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
class MultiSensor_DecomposeWHAR_v1(nn.Module):
    def __init__(self, 
                 imu_num_sensor=cfg.imu_num_sensor, imu_M=7,
                 thm_num_sensor=cfg.thm_num_sensor, thm_M=1,
                 tof_num_sensor=cfg.tof_num_sensor, tof_M=64,
                 L=cfg.seq_len, D=cfg.ddim, num_classes=cfg.num_classes, 
                 S=4, use_cross_sensor=True):
        super().__init__()
        
        self.S = S
        self.use_cross_sensor = use_cross_sensor
        
        self.imu_dwhar = DecomposeWHAR_Extractor(
            num_sensor=imu_num_sensor, M=imu_M, L=L, D=D, S=S
        )
        self.thm_dwhar = DecomposeWHAR_Extractor(
            num_sensor=thm_num_sensor, M=thm_M, L=L, D=D, S=S
        )
        self.tof_dwhar = DecomposeWHAR_Extractor(
            num_sensor=tof_num_sensor, M=tof_M, L=L, D=D, S=S
        )
        
        imu_size = imu_num_sensor * D * (L // S)
        thm_size = thm_num_sensor * D * (L // S)
        tof_size = tof_num_sensor * D * (L // S)
        
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

# stupid 1-sensor dwhar
class DecomposeWHAR_SingleSensor_v1(nn.Module):
    def __init__(
            self, 
            M=len(cfg.imu_cols),
            L=cfg.seq_len,
            num_classes=cfg.num_classes,
            D=cfg.ddim,
            S=cfg.stride,
        ):
        super().__init__()

        self.model = DecomposeWHAR_Extractor(
            num_sensor=1,
            M=M,
            L=L,
            D=D,
            S=S
        )

        self.classifier = nn.Linear(D * (L // S), num_classes)
    
    def forward(self, data):
        x = self.model(data)
        pred = self.classifier(x)
        return pred

# multi-sensor w/ diff embedding layers and backbones
class MultiSensor_DecomposeWHAR_v2(nn.Module):
    def __init__(self,
                 num_imu=cfg.imu_num_sensor, 
                 num_tof=cfg.tof_num_sensor,
                 num_thm=cfg.thm_num_sensor, 
                 imu_vars=7, 
                 tof_vars=64,  
                 thm_vars=1,   
                 L=cfg.seq_len,  
                 D=cfg.ddim,   
                 P=cfg.emb_kernel_size, 
                 S=cfg.stride,  
                 kernel_size=cfg.kernel_size,
                 r=cfg.reduction_ratio,
                 num_layers=cfg.num_layers, 
                 num_m_layers=cfg.num_m_layers, 
                 num_a_layers=cfg.num_a_layers,
                 num_classes=cfg.num_classes):
        super(MultiSensor_DecomposeWHAR_v2, self).__init__()
        
        self.num_imu = num_imu
        self.num_tof = num_tof  
        self.num_thm = num_thm
        self.imu_vars = imu_vars
        self.tof_vars = tof_vars
        self.thm_vars = thm_vars
        self.D = D
        self.num_layers = num_layers
        self.num_m_layers = num_m_layers
        self.num_a_layers = num_a_layers
        
        T = L // S
        self.T = T
        
        self.imu_embed = Embedding(P, S, D)
        self.tof_embed = ToFEmbedding(P, S, D)
        self.thm_embed = Embedding(P, S, D)
        
        self.imu_backbone = nn.ModuleList([
            DecomposeConvBlock(imu_vars, D, kernel_size, r) 
            for _ in range(num_layers)
        ])
        
        self.tof_backbone = nn.ModuleList([
            DecomposeConvBlock(16, D, kernel_size, r) # 16 after 2d-conv
            for _ in range(num_layers)
        ])
        
        self.thm_backbone = nn.ModuleList([
            DecomposeConvBlock(thm_vars, D, kernel_size, r)
            for _ in range(num_layers)
        ])
        
        total_sensors = num_imu + num_tof + num_thm # 11
        d_model_mamba = total_sensors * D
        
        self.mamba_preprocess = nn.ModuleList([
            Mamba_Layer(Mamba(d_model=d_model_mamba, d_state=16, d_conv=4), d_model_mamba)
            for _ in range(num_m_layers)
        ])
        
        self.AM_layers = nn.ModuleList([
            Att_Layer(
                AttentionLayer(
                    FullAttention(False, 1, attention_dropout=0.05, output_attention=True),
                    D * T, 8),
                D * T, 0.05
            ) for _ in range(num_a_layers)
        ])
        
        self.fc_out = nn.Linear(total_sensors * D * T, num_classes)
        self.dropout_prob = 0.6

    def forward(self, imu_data, thm_data, tof_data):
        # inputs: 'imu', 'tof', 'thm'
        # imu: [B, 1, L, 7]
        # tof: [B, 5, L, 64] 
        # thm: [B, 5, L, 1]
        
        B = imu_data.shape[0]
        processed_sensors = []
        
        imu_x = imu_data.reshape(B * self.num_imu, imu_data.shape[2], imu_data.shape[3])
        imu_x = imu_x.permute(0, 2, 1)  # [B*1, 7, L]
        imu_emb = self.imu_embed(imu_x)  # [B*1, 7, D, T]
        
        for layer in self.imu_backbone:
            imu_emb = layer(imu_emb)
        
        imu_emb = rearrange(imu_emb, 'b m d t -> b m (d t)', b=B*self.num_imu)
        imu_emb = imu_emb.mean(dim=1)  # [B*1, D*T]
        imu_emb = imu_emb.reshape(B, self.num_imu, -1)  # [B, 1, D*T]
        processed_sensors.append(imu_emb)
        
        tof_data = tof_data  # [B, 5, L, 64]
        tof_x = tof_data.reshape(B * self.num_tof, tof_data.shape[2], tof_data.shape[3])
        tof_x = tof_x.permute(0, 2, 1)  # [B*5, 64, L]
        tof_emb = self.tof_embed(tof_x)  # [B*5, 16, D, T] после 2D обработки
        
        for layer in self.tof_backbone:
            tof_emb = layer(tof_emb)
            
        tof_emb = rearrange(tof_emb, 'b m d t -> b m (d t)', b=B*self.num_tof)
        tof_emb = tof_emb.mean(dim=1)  # [B*5, D*T]
        tof_emb = tof_emb.reshape(B, self.num_tof, -1)  # [B, 5, D*T]
        processed_sensors.append(tof_emb)
        
        thm_data = thm_data  # [B, 5, L, 1]
        thm_x = thm_data.reshape(B * self.num_thm, thm_data.shape[2], thm_data.shape[3])
        thm_x = thm_x.permute(0, 2, 1)  # [B*5, 1, L]
        thm_emb = self.thm_embed(thm_x)  # [B*5, 1, D, T]
        
        for layer in self.thm_backbone:
            thm_emb = layer(thm_emb)
            
        thm_emb = rearrange(thm_emb, 'b m d t -> b m (d t)', b=B*self.num_thm)
        thm_emb = thm_emb.mean(dim=1)  # [B*5, D*T]
        thm_emb = thm_emb.reshape(B, self.num_thm, -1)  # [B, 5, D*T]
        processed_sensors.append(thm_emb)
        
        x_emb = torch.cat(processed_sensors, dim=1)  # [B, 11, D*T]
        
        x_emb = x_emb.reshape(B, x_emb.shape[1], self.D, self.T)  # [B, 11, D, T]
        x_emb = x_emb.permute(0, 3, 1, 2)  # [B, T, 11, D]
        x_emb = x_emb.reshape(B, x_emb.shape[1], -1)  # [B, T, 11*D]
        
        for layer in self.mamba_preprocess:
            x_emb = layer(x_emb)
            
        x_emb = x_emb.reshape(B, x_emb.shape[1], 11, self.D)  # [B, T, 11, D]
        x_emb = x_emb.permute(0, 2, 3, 1)  # [B, 11, D, T]
        x_emb = x_emb.reshape(B, x_emb.shape[1], -1)  # [B, 11, D*T]
        
        for layer in self.AM_layers:
            x_emb, _ = layer(x_emb, None)
            
        x = x_emb.reshape(B, -1)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        pred = self.fc_out(x)
        
        return x, pred

# simplified dwhar for 1-sensor input
class DecomposeWHAR_SingleSensor_v2(nn.Module):
    def __init__(self,
                 M,  # Number of variables in the multivariate sequence
                 L,  # Length of the input sequence (time steps)
                 D=cfg.ddim,  # Number of channels per variable
                 P=cfg.emb_kernel_size,  # Kernel size of the embedding layer
                 S=cfg.stride,  # Stride of the embedding layer
                 kernel_size=cfg.kernel_size,  # Kernel size for convolutional layers
                 r=cfg.reduction_ratio,  # A hyperparameter for decomposition (e.g., reduction ratio)
                 num_layers=cfg.num_layers,  # Number of decomposition layers
                 num_m_layers=cfg.num_m_layers,  # Number of mamba layers
                 num_classes=cfg.num_classes):   
        super(DecomposeWHAR_SingleSensor_v2, self).__init__()

        self.num_layers = num_layers
        self.num_m_layers = num_m_layers
        T = L // S  # Calculate the number of patches after embedding
        self.T = T
        self.D = D
        self.M = M
        
        # Embedding layer to transform input sequences into higher dimensional representations
        self.embed_layer = Embedding(P, S, D)
        
        # Backbone consisting of multiple decomposition convolutional blocks
        self.backbone = nn.ModuleList([DecomposeConvBlock(M, D, kernel_size, r) for _ in range(num_layers)])
        
        self.dropout_prob = 0.6  # Dropout probability

        d_model_mamba = D  # Model dimension for Mamba (single sensor)
        d_state = 16  # State dimension for Mamba
        d_conv = 4  # Convolutional dimension for Mamba

        # Mamba Block for Global Temporal Aggregation (GTA)
        self.mamba_preprocess = nn.ModuleList([
            Mamba_Layer(Mamba(d_model=d_model_mamba, d_state=d_state, d_conv=d_conv), d_model_mamba)
            for _ in range(num_m_layers)
        ])

        # Final classification layer
        self.fc_out = nn.Linear(1 * D * T, num_classes)

    def forward(self, inputs):  # inputs: (B, 1, L, M) - Batch size, 1 sensor, Sequence length, Number of variables
        B, N, L, M = inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3] # bs,n_sensors,seq_len,n_vars
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

        x = x_emb

        x = x.reshape(inputs.shape[0],-1)      

        # Dropout and classification
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        pred = self.fc_out(x)  # [B, D*T] -> [B, num_classes]

        return x, pred  # output: [B, num_classes]