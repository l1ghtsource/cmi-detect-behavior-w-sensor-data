import torch
import torch.nn as nn
import numpy as np
from modules.inceptiontime import InceptionTimeFeatureExtractor
from modules.nystrom_attention import NystromAttention
from modules.statistic_extractor import StatisticExtractor
from configs.config import cfg

# paper -> https://arxiv.org/pdf/2405.03140 !!

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dropout=0.2,dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm(x), mask=mask)
        return x
    
### Define Wavelet Kernel
def mexican_hat_wavelet(size, scale, shift): #size :d*kernelsize  scale:d*1 shift:d*1
    """
    Generate a Mexican Hat wavelet kernel.

    Parameters:
    size (int): Size of the kernel.
    scale (float): Scale of the wavelet.
    shift (float): Shift of the wavelet.

    Returns:
    torch.Tensor: Mexican Hat wavelet kernel.
    """
  
    x = torch.linspace(-( size[1]-1)//2, ( size[1]-1)//2, size[1]).cuda()
    # print(x.shape)
    x = x.reshape(1,-1).repeat(size[0],1)
    # print(x.shape)
    # print(shift.shape)
    x = x - shift  # Apply the shift

    # Mexican Hat wavelet formula
    C = 2 / ( 3**0.5 * torch.pi**0.25)
    wavelet = C * (1 - (x/scale)**2) * torch.exp(-(x/scale)**2 / 2)*1  /(torch.abs(scale)**0.5)

    return wavelet #d*L

class WaveletEncoding(nn.Module):
    def __init__(self, dim=512, max_len = 256,hidden_len = 512,dropout=0.0):
        super().__init__()
       
        #n_w =3
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_3 = nn.Linear(dim, dim)
        
    def forward(self,x,wave1,wave2,wave3):
        cls_token, feat_token = x[:, 0], x[:, 1:]
        
        x = feat_token.transpose(1, 2)
        
        D = x.shape[1]
        scale1, shift1 =wave1[0,:],wave1[1,:]
        wavelet_kernel1 = mexican_hat_wavelet(size=(D,19), scale=scale1, shift=shift1)
        scale2, shift2 =wave2[0,:],wave2[1,:]
        wavelet_kernel2 = mexican_hat_wavelet(size=(D,19), scale=scale2, shift=shift2)
        scale3, shift3 =wave3[0,:],wave3[1,:]
        wavelet_kernel3 = mexican_hat_wavelet(size=(D,19), scale=scale3, shift=shift3)
        
        #Eq. 11
        pos1= torch.nn.functional.conv1d(x,wavelet_kernel1.unsqueeze(1),groups=D,padding ='same')
        pos2= torch.nn.functional.conv1d(x,wavelet_kernel2.unsqueeze(1),groups=D,padding ='same')
        pos3= torch.nn.functional.conv1d(x,wavelet_kernel3.unsqueeze(1),groups=D,padding ='same')
        x = x.transpose(1, 2)   #B*N*D
        # print(x.shape)

        #Eq. 10
        x = x + self.proj_1(pos1.transpose(1, 2)+pos2.transpose(1, 2)+pos3.transpose(1, 2))# + mixup_encording
        
        # mixup token information
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class Original_TimeMIL(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,dropout=0.):
        super().__init__()
    
        # define backbone Can be replace here
        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features )
          
        # define WPE    
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.wave1 = torch.randn(2, mDim,1 )
        self.wave1[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 )  #make sure scale >0
        self.wave1 = nn.Parameter(self.wave1)
        
        self.wave2 = torch.zeros(2, mDim,1 )
        self.wave2[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2 = nn.Parameter(self.wave2)
        
        self.wave3 = torch.zeros(2, mDim,1 )
        self.wave3[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3 = nn.Parameter(self.wave3)    
            
        self.wave1_ = torch.randn(2, mDim,1 )
        self.wave1_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave1_ = nn.Parameter(self.wave1)
        
        self.wave2_ = torch.zeros(2, mDim,1 )
        self.wave2_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2_ = nn.Parameter(self.wave2)
        
        self.wave3_ = torch.zeros(2, mDim,1 )
        self.wave3_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3_ = nn.Parameter(self.wave3)        
            
        hidden_len = 2* max_seq_len
            
        # define class token      
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.pos_layer =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        self.pos_layer2 =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        # self.pos_layer = ConvPosEncoding1D(mDim)
        self.layer1 = TransLayer(dim=mDim,dropout=dropout)
        self.layer2 = TransLayer(dim=mDim,dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        # self._fc2 = nn.Linear(mDim, n_classes)
        self._fc2 = nn.Sequential(
            nn.Linear(mDim,mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        ) 
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        initialize_weights(self)
        
        
    def forward(self, x,warmup=False):
        x1 = self.feature_extractor(x.transpose(1, 2))
        x1 = x1.transpose(1, 2)
        x= x1

        B, seq_len, D = x.shape

        view_x = x.clone()
        
        global_token = x.mean(dim=1)#[0]
        
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)    #B * 1 * d
      
        x = torch.cat((cls_tokens, x), dim=1)
        # WPE1
        x = self.pos_layer(x,self.wave1,self.wave2,self.wave3)
      
        # TransLayer x1
        x = self.layer1(x)
        # WPE2
        x = self.pos_layer2(x,self.wave1_,self.wave2_,self.wave3_)
       
        # TransLayer x2
        x = self.layer2(x)
       
        # only cls_token is used for cls
        x = x[:,0]

        # stablity of training random initialized global token
        if warmup:
            x = 0.1*x+0.99*global_token   
 
        logits = self._fc2(x)
            
        return logits

# for imu + tof + thm !! lesssgo
class MultiSensor_TimeMIL_v1(nn.Module):
    def __init__(self, n_classes=cfg.num_classes, mDim=cfg.timemil_dim, max_seq_len=cfg.seq_len, dropout=cfg.timemil_dropout):
        super().__init__()
     
        # Define separate feature extractors for each sensor type
        self.imu_feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=cfg.imu_vars)
        self.tof_feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=cfg.tof_vars)  
        self.thm_feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=cfg.thm_vars) 

        # Statistics extractor for computing features on-the-fly
        self.statistic_extractor = StatisticExtractor()
        
        # Stats processors for each sensor type
        if cfg.use_stats_vectors:
            self.imu_stats_processor = nn.Sequential(
                nn.Linear(cfg.imu_vars * 10, 128),  # 10 stats per IMU channel
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64)
            )
            
            self.tof_stats_processor = nn.Sequential(
                nn.Linear(cfg.tof_vars * 10, 512),  # 10 stats per ToF channel  
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 64)
            )
            
            self.thm_stats_processor = nn.Sequential(
                nn.Linear(cfg.thm_vars * 10, 128),  # 10 stats per thermal channel
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64)
            )

        # Calculate total features
        base_features = (cfg.imu_num_sensor + cfg.tof_num_sensor + cfg.thm_num_sensor) * 128
        stats_features = 0
        if cfg.use_stats_vectors:
            stats_features = (cfg.imu_num_sensor + cfg.tof_num_sensor + cfg.thm_num_sensor) * 64
        
        total_features = base_features + stats_features
        
        # Projection layer to map concatenated features to target dimension
        self.feature_proj = nn.Linear(total_features, mDim)
        
        # Late fusion projections for stats (if enabled)
        if cfg.use_stats_vectors:
            self.imu_late_fusion_proj = nn.Linear(64, mDim)
            self.tof_late_fusion_proj = nn.Linear(64, mDim)
            self.thm_late_fusion_proj = nn.Linear(64, mDim)
            
        # Define WPE parameters    
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.wave1 = torch.randn(2, mDim, 1)
        self.wave1[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # ensure scale > 0
        self.wave1 = nn.Parameter(self.wave1)
        
        self.wave2 = torch.zeros(2, mDim, 1)
        self.wave2[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave2 = nn.Parameter(self.wave2)
        
        self.wave3 = torch.zeros(2, mDim, 1)
        self.wave3[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave3 = nn.Parameter(self.wave3)    
            
        self.wave1_ = torch.randn(2, mDim, 1)
        self.wave1_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave1_ = nn.Parameter(self.wave1_)
        
        self.wave2_ = torch.zeros(2, mDim, 1)
        self.wave2_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave2_ = nn.Parameter(self.wave2_)
        
        self.wave3_ = torch.zeros(2, mDim, 1)
        self.wave3_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave3_ = nn.Parameter(self.wave3_)        
            
        hidden_len = 2 * max_seq_len
            
        # Define positional encoding and transformer layers      
        self.pos_layer = WaveletEncoding(mDim, max_seq_len, hidden_len) 
        self.pos_layer2 = WaveletEncoding(mDim, max_seq_len, hidden_len) 
        self.layer1 = TransLayer(dim=mDim, dropout=dropout)
        self.layer2 = TransLayer(dim=mDim, dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        
        # Final classification layers
        if cfg.use_stats_vectors:
            final_dim = mDim * 2  # cls_token + mean sensor stats
        else:
            final_dim = mDim
            
        self._fc2 = nn.Sequential(
            nn.Linear(final_dim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        ) 

        self._fc2_aux2 = nn.Sequential(
            nn.Linear(final_dim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, 2),
        )
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        initialize_weights(self)

    def apply_feature_mask(self, features, mask):
        # mask is [B, L], expand to match features
        mask = mask.unsqueeze(1)  # [B, 1, L]
        return features * mask.float()
        
    def forward(self, imu_data, thm_data, tof_data, pad_mask=None, warmup=False):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data  
            thm_data: [B, 5, L, 1] - Thermal sensor data
            pad_mask: [B, L] - padding mask (1=valid, 0=padding) [optional!]
            warmup: bool - whether to use warmup strategy [optional!]
        """
        B, _, L, _ = imu_data.shape
        
        # Compute statistics if enabled
        imu_stats_processed = None
        tof_stats_processed = None
        thm_stats_processed = None
        
        if cfg.use_stats_vectors:
            imu_raw = imu_data[:, 0, :, :].transpose(1, 2)  # [B, 7, L]
            imu_stats = self.statistic_extractor(imu_raw, pad_mask)  # [B, 7*10]
            imu_stats_processed = self.imu_stats_processor(imu_stats)  # [B, 64]
            
            tof_raw = tof_data.transpose(2, 3).contiguous().view(B, -1, L)  # [B, 5*64, L]
            tof_stats = self.statistic_extractor(tof_raw, pad_mask)  # [B, 5*64*10]
            tof_stats_processed = self.tof_stats_processor(tof_stats)  # [B, 64]
            
            thm_raw = thm_data.transpose(2, 3).contiguous().view(B, -1, L)  # [B, 5*1, L]
            thm_stats = self.statistic_extractor(thm_raw, pad_mask)  # [B, 5*1*10]
            thm_stats_processed = self.thm_stats_processor(thm_stats)  # [B, 64]
        
        # Process each sensor type and collect features
        all_features = []
        
        # Process IMU data (1 sensor)
        for i in range(cfg.imu_num_sensor):
            sensor_data = imu_data[:, i, :, :].transpose(1, 2)  # [B, 7, L]
            features = self.imu_feature_extractor(sensor_data)  # [B, 128, L]
            if pad_mask is not None:
                features = self.apply_feature_mask(features, pad_mask)
            all_features.append(features)
            
            # Add stats features for early fusion
            if cfg.use_stats_vectors:
                stats_expanded = imu_stats_processed.unsqueeze(2).expand(-1, -1, L)  # [B, 64, L]
                all_features.append(stats_expanded)
        
        # Process ToF data (5 sensors)
        for i in range(cfg.tof_num_sensor):
            sensor_data = tof_data[:, i, :, :].transpose(1, 2)  # [B, 64, L]
            features = self.tof_feature_extractor(sensor_data)  # [B, 128, L]
            if pad_mask is not None:
                features = self.apply_feature_mask(features, pad_mask)
            all_features.append(features)
            
            # Add stats features for early fusion
            if cfg.use_stats_vectors:
                stats_expanded = tof_stats_processed.unsqueeze(2).expand(-1, -1, L)  # [B, 64, L]
                all_features.append(stats_expanded)
        
        # Process Thermal data (5 sensors)
        for i in range(cfg.thm_num_sensor):
            sensor_data = thm_data[:, i, :, :].transpose(1, 2)  # [B, 1, L]
            features = self.thm_feature_extractor(sensor_data)  # [B, 128, L]
            if pad_mask is not None:
                features = self.apply_feature_mask(features, pad_mask)
            all_features.append(features)
            
            # Add stats features for early fusion
            if cfg.use_stats_vectors:
                stats_expanded = thm_stats_processed.unsqueeze(2).expand(-1, -1, L)  # [B, 64, L]
                all_features.append(stats_expanded)
        
        # Concatenate all features along channel dimension
        x = torch.cat(all_features, dim=1)  # [B, total_features, L]
        
        # Transpose and project to target dimension
        x = x.transpose(1, 2)  # [B, L, total_features]
        x = self.feature_proj(x)  # [B, L, mDim]

        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).float()  # [B, L, mDim]

        # Compute global token
        if pad_mask is not None:
            valid_counts = pad_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
            valid_counts = torch.clamp(valid_counts, min=1.0)  # Avoid division by zero
            global_token = (x * pad_mask.unsqueeze(-1).float()).sum(dim=1) / valid_counts  # [B, mDim]
        else:
            global_token = x.mean(dim=1)  # [B, mDim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, mDim]
        x = torch.cat((cls_tokens, x), dim=1)

        if pad_mask is not None:
            extended_mask = torch.cat([
                torch.ones(B, 1, device=pad_mask.device, dtype=torch.bool),  # cls token is always valid
                pad_mask
            ], dim=1)  # [B, L+1]
        
        # Apply Wavelet Positional Encoding 1
        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3)
        
        # Apply TransLayer 1
        if pad_mask is not None:
            x = self.layer1(x, mask=extended_mask.bool())
        else:
            x = self.layer1(x)
        
        # Apply Wavelet Positional Encoding 2
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_)
        
        # Apply TransLayer 2
        if pad_mask is not None:
            x = self.layer2(x, mask=extended_mask.bool())
        else:
            x = self.layer2(x)
        
        # Extract class token
        cls_token_output = x[:, 0]
        
        # Late fusion with stats
        if cfg.use_stats_vectors:
            imu_stats_late = self.imu_late_fusion_proj(imu_stats_processed)  # [B, mDim]
            tof_stats_late = self.tof_late_fusion_proj(tof_stats_processed)  # [B, mDim]
            thm_stats_late = self.thm_late_fusion_proj(thm_stats_processed)  # [B, mDim]
            
            final_output = torch.cat([
                cls_token_output, 
                (imu_stats_late + tof_stats_late + thm_stats_late) / 3
            ], dim=1)  # [B, 2*mDim]
        else:
            final_output = cls_token_output
        
        # Apply warmup strategy if needed
        if warmup:
            if cfg.use_stats_vectors:
                global_token_extended = torch.cat([
                    global_token, 
                    torch.zeros_like(global_token),
                    torch.zeros_like(global_token),
                    torch.zeros_like(global_token)
                ], dim=1)
                final_output = 0.1 * final_output + 0.99 * global_token_extended
            else:
                final_output = 0.1 * final_output + 0.99 * global_token   
 
        # Final classification
        logits = self._fc2(final_output)
        logits_aux2 = self._fc2_aux2(final_output)
            
        return logits, logits_aux2
            
# ya ebal eto govnishe
class TimeMIL_SingleSensor_v1(nn.Module):
    def __init__(self, n_classes=cfg.num_classes, mDim=cfg.timemil_dim, max_seq_len=cfg.seq_len, dropout=cfg.timemil_dropout):
        super().__init__()
     
        # Define feature extractor for IMU sensor only
        self.imu_feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=cfg.imu_vars)

        # Statistics extractor for computing features on-the-fly
        self.statistic_extractor = StatisticExtractor()
        
        # Stats processor for IMU statistics
        if cfg.use_stats_vectors:
            self.imu_stats_processor = nn.Sequential(
                nn.Linear(cfg.imu_vars * 10, 128),  # 10 stats per IMU channel (7*10=70)
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64)
            )
        
        # Calculate total features for early fusion
        base_features = 128  # InceptionTime output
        if cfg.use_stats_vectors:
            total_features = base_features + 64  # 128 + 64 = 192
        else:
            total_features = base_features  # 128
        
        # Projection layer to map features to target dimension
        self.feature_proj = nn.Linear(total_features, mDim)
        
        # Late fusion projection for stats
        if cfg.use_stats_vectors:
            self.imu_late_fusion_proj = nn.Linear(64, mDim)
            
        # Define WPE parameters    
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.wave1 = torch.randn(2, mDim, 1)
        self.wave1[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # ensure scale > 0
        self.wave1 = nn.Parameter(self.wave1)
        
        self.wave2 = torch.zeros(2, mDim, 1)
        self.wave2[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave2 = nn.Parameter(self.wave2)
        
        self.wave3 = torch.zeros(2, mDim, 1)
        self.wave3[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave3 = nn.Parameter(self.wave3)    
            
        self.wave1_ = torch.randn(2, mDim, 1)
        self.wave1_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave1_ = nn.Parameter(self.wave1_)
        
        self.wave2_ = torch.zeros(2, mDim, 1)
        self.wave2_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave2_ = nn.Parameter(self.wave2_)
        
        self.wave3_ = torch.zeros(2, mDim, 1)
        self.wave3_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        self.wave3_ = nn.Parameter(self.wave3_)        
            
        hidden_len = 2 * max_seq_len
            
        # Define positional encoding and transformer layers      
        self.pos_layer = WaveletEncoding(mDim, max_seq_len, hidden_len) 
        self.pos_layer2 = WaveletEncoding(mDim, max_seq_len, hidden_len) 
        self.layer1 = TransLayer(dim=mDim, dropout=dropout)
        self.layer2 = TransLayer(dim=mDim, dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        
        # Final classification layers (late fusion: cls_token + imu_stats = 2 * mDim)
        if cfg.use_stats_vectors:
            final_dim = mDim * 2  # cls_token + imu_stats
        else:
            final_dim = mDim
            
        self._fc2 = nn.Sequential(
            nn.Linear(final_dim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        ) 

        self._fc2_aux2 = nn.Sequential(
            nn.Linear(final_dim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, 2),
        )
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        initialize_weights(self)

    def apply_feature_mask(self, features, mask):
        # mask is [B, L], expand to match features
        mask = mask.unsqueeze(1)  # [B, 1, L]
        return features * mask.float()
        
    def forward(self, imu_data, pad_mask=None, warmup=False):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            pad_mask: [B, L] - padding mask (1=valid, 0=padding) [optional!]
            warmup: bool - whether to use warmup strategy [optional!]
        """
        B, _, L, _ = imu_data.shape
        
        # Process IMU data (single sensor)
        sensor_data = imu_data[:, 0, :, :].transpose(1, 2)  # [B, 7, L]
        temporal_features = self.imu_feature_extractor(sensor_data)  # [B, 128, L]

        if pad_mask is not None:
            temporal_features = self.apply_feature_mask(temporal_features, pad_mask)
        
        # Compute statistics if enabled
        imu_stats_processed = None
        if cfg.use_stats_vectors:
            # Extract IMU statistics
            imu_stats = self.statistic_extractor(sensor_data, pad_mask)  # [B, 7*10=70]
            imu_stats_processed = self.imu_stats_processor(imu_stats)  # [B, 64]
            
            # Early fusion: expand stats to temporal dimension and concatenate
            stats_expanded = imu_stats_processed.unsqueeze(2).expand(-1, -1, L)  # [B, 64, L]
            x = torch.cat([temporal_features, stats_expanded], dim=1)  # [B, 192, L]
        else:
            x = temporal_features  # [B, 128, L]
        
        # Transpose and project to target dimension
        x = x.transpose(1, 2)  # [B, L, 192] or [B, L, 128]
        x = self.feature_proj(x)  # [B, L, mDim]

        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).float()  # [B, L, mDim]

        # Compute global token
        if pad_mask is not None:
            valid_counts = pad_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
            valid_counts = torch.clamp(valid_counts, min=1.0)  # Avoid division by zero
            global_token = (x * pad_mask.unsqueeze(-1).float()).sum(dim=1) / valid_counts  # [B, mDim]
        else:
            global_token = x.mean(dim=1)  # [B, mDim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, mDim]
        x = torch.cat((cls_tokens, x), dim=1)

        if pad_mask is not None:
            extended_mask = torch.cat([
                torch.ones(B, 1, device=pad_mask.device, dtype=torch.bool),  # cls token is always valid
                pad_mask
            ], dim=1)  # [B, L+1]
        
        # Apply Wavelet Positional Encoding 1
        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3)
        
        # Apply TransLayer 1
        if pad_mask is not None:
            x = self.layer1(x, mask=extended_mask.bool())
        else:
            x = self.layer1(x)
        
        # Apply Wavelet Positional Encoding 2
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_)
        
        # Apply TransLayer 2
        if pad_mask is not None:
            x = self.layer2(x, mask=extended_mask.bool())
        else:
            x = self.layer2(x)
        
        # Extract class token
        cls_token_output = x[:, 0]  # [B, mDim]
        
        # Late fusion with stats (if enabled)
        if cfg.use_stats_vectors:
            imu_stats_late = self.imu_late_fusion_proj(imu_stats_processed)  # [B, mDim]
            final_output = torch.cat([cls_token_output, imu_stats_late], dim=1)  # [B, 2*mDim]
        else:
            final_output = cls_token_output  # [B, mDim]
        
        # Apply warmup strategy if needed
        if warmup:
            if cfg.use_stats_vectors:
                # Extend global_token to match fused_output dimension
                global_token_extended = torch.cat([
                    global_token, 
                    torch.zeros_like(global_token)
                ], dim=1)  # [B, 2*mDim]
                final_output = 0.1 * final_output + 0.99 * global_token_extended
            else:
                final_output = 0.1 * final_output + 0.99 * global_token   
 
        # Final classification
        logits = self._fc2(final_output)
        logits_aux2 = self._fc2_aux2(final_output)
            
        return logits, logits_aux2
    
class SensorProcessor(nn.Module):
    def __init__(self, feature_extractor, mDim, max_seq_len, num_sensors):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.proj = nn.Linear(num_sensors * 128, mDim)
        
        hidden_len = 2 * max_seq_len
        self.pos_layer1 = WaveletEncoding(mDim, max_seq_len, hidden_len)
        self.pos_layer2 = WaveletEncoding(mDim, max_seq_len, hidden_len)
        
        self.wave1 = nn.Parameter(torch.randn(2, mDim, 1))
        self.wave1.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        
        self.wave2 = nn.Parameter(torch.zeros(2, mDim, 1))
        self.wave2.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        
        self.wave3 = nn.Parameter(torch.zeros(2, mDim, 1))
        self.wave3.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        
        self.wave1_ = nn.Parameter(torch.randn(2, mDim, 1))
        self.wave1_.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        
        self.wave2_ = nn.Parameter(torch.zeros(2, mDim, 1))
        self.wave2_.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)
        
        self.wave3_ = nn.Parameter(torch.zeros(2, mDim, 1))
        self.wave3_.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)

class CrossAttentionFusion(nn.Module):
    def __init__(self, mDim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mDim = mDim
        self.num_heads = num_heads
        
        self.imu_tof_cross_attn = nn.MultiheadAttention(mDim, num_heads, dropout=dropout, batch_first=True)
        self.imu_thm_cross_attn = nn.MultiheadAttention(mDim, num_heads, dropout=dropout, batch_first=True)
        self.tof_thm_cross_attn = nn.MultiheadAttention(mDim, num_heads, dropout=dropout, batch_first=True)
        
        self.fusion_proj = nn.Linear(mDim * 3, mDim)
        self.norm = nn.LayerNorm(mDim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, imu_features, tof_features, thm_features, mask=None):
        imu_tof_out, _ = self.imu_tof_cross_attn(
            imu_features, tof_features, tof_features, 
            key_padding_mask=mask if mask is not None else None
        )
        
        imu_thm_out, _ = self.imu_thm_cross_attn(
            imu_features, thm_features, thm_features,
            key_padding_mask=mask if mask is not None else None
        )
        
        tof_thm_out, _ = self.tof_thm_cross_attn(
            tof_features, thm_features, thm_features,
            key_padding_mask=mask if mask is not None else None
        )
        
        fused_features = torch.cat([imu_tof_out, imu_thm_out, tof_thm_out], dim=-1)
        fused_features = self.fusion_proj(fused_features)
        
        fused_features = self.norm(fused_features + imu_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features

class MultiSensor_TimeMIL_v2(nn.Module):
    def __init__(self, n_classes=cfg.num_classes, mDim=cfg.timemil_dim, max_seq_len=cfg.seq_len, dropout=cfg.timemil_dropout):
        super().__init__()
        
        # Statistics extractor for computing features on-the-fly
        self.statistic_extractor = StatisticExtractor()
        
        # Stats processors for each sensor type
        if cfg.use_stats_vectors:
            self.imu_stats_processor = nn.Sequential(
                nn.Linear(cfg.imu_vars * 10, 128),  # 10 stats per IMU channel
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64)
            )
            
            self.tof_stats_processor = nn.Sequential(
                nn.Linear(cfg.tof_vars * cfg.tof_num_sensor * 10, 512),  # 10 stats per ToF channel per sensor
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 64)
            )
            
            self.thm_stats_processor = nn.Sequential(
                nn.Linear(cfg.thm_vars * cfg.thm_num_sensor * 10, 128),  # 10 stats per thermal channel per sensor
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64)
            )
            
            # Late fusion projections for stats
            self.imu_late_fusion_proj = nn.Linear(64, mDim)
            self.tof_late_fusion_proj = nn.Linear(64, mDim)
            self.thm_late_fusion_proj = nn.Linear(64, mDim)
        
        self.imu_processor = SensorProcessor(
            InceptionTimeFeatureExtractor(n_in_channels=cfg.imu_vars),
            mDim, max_seq_len, cfg.imu_num_sensor
        )
        
        self.tof_processor = SensorProcessor(
            InceptionTimeFeatureExtractor(n_in_channels=cfg.tof_vars),
            mDim, max_seq_len, cfg.tof_num_sensor
        )
        
        self.thm_processor = SensorProcessor(
            InceptionTimeFeatureExtractor(n_in_channels=cfg.thm_vars),
            mDim, max_seq_len, cfg.thm_num_sensor
        )
        
        # Update CrossAttentionFusion for potentially augmented features
        if cfg.use_stats_vectors:
            # If using stats, we'll modify features before fusion
            self.cross_attention_fusion = CrossAttentionFusion(mDim + 64, num_heads=8, dropout=dropout)
        else:
            self.cross_attention_fusion = CrossAttentionFusion(mDim, num_heads=8, dropout=dropout)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        
        self.layer1 = TransLayer(dim=mDim, dropout=dropout)
        self.layer2 = TransLayer(dim=mDim, dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        
        # Final classification layers (late fusion: cls_token + mean stats = 2 * mDim)
        if cfg.use_stats_vectors:
            final_dim = mDim * 2  # cls_token + mean_proj(imu_stats + tof_stats + thm_stats)
        else:
            final_dim = mDim
            
        self._fc2 = nn.Sequential(
            nn.Linear(final_dim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        )
        
        self._fc2_aux2 = nn.Sequential(
            nn.Linear(final_dim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, 2),
        )
        
        self.alpha = nn.Parameter(torch.ones(1))
        initialize_weights(self)
    
    def process_sensor_data(self, processor, sensor_data, sensor_stats=None, pad_mask=None):
        B, num_sensors, L, channels = sensor_data.shape
        all_features = []
        
        for i in range(num_sensors):
            sensor_features = sensor_data[:, i, :, :].transpose(1, 2)  # [B, channels, L]
            features = processor.feature_extractor(sensor_features)  # [B, 128, L]
            
            if pad_mask is not None:
                features = features * pad_mask.unsqueeze(1).float()
            
            all_features.append(features)
        
        x = torch.cat(all_features, dim=1)  # [B, num_sensors*128, L]
        x = x.transpose(1, 2)  # [B, L, num_sensors*128]
        x = processor.proj(x)  # [B, L, mDim]
        
        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).float()
        
        # Apply positional encoding
        x = processor.pos_layer1(x, processor.wave1, processor.wave2, processor.wave3)
        x = processor.pos_layer2(x, processor.wave1_, processor.wave2_, processor.wave3_)
        
        # Early fusion with stats if provided
        if cfg.use_stats_vectors and sensor_stats is not None:
            # Expand stats to temporal dimension
            stats_expanded = sensor_stats.unsqueeze(1).expand(-1, L, -1)  # [B, L, 64]
            x = torch.cat([x, stats_expanded], dim=2)  # [B, L, mDim+64]
        
        return x
    
    def forward(self, imu_data, thm_data, tof_data, pad_mask=None, warmup=False):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data  
            thm_data: [B, 5, L, 1] - Thermal sensor data
            pad_mask: [B, L] - padding mask (1=valid, 0=padding)
            warmup: bool - whether to use warmup strategy
        """
        B, _, L, _ = imu_data.shape
        
        # Compute statistics if enabled
        imu_stats_processed = None
        tof_stats_processed = None
        thm_stats_processed = None
        
        if cfg.use_stats_vectors:
            # Extract IMU statistics
            imu_raw = imu_data[:, 0, :, :].transpose(1, 2)  # [B, 7, L]
            imu_stats = self.statistic_extractor(imu_raw, pad_mask)  # [B, 7*10]
            imu_stats_processed = self.imu_stats_processor(imu_stats)  # [B, 64]
            
            # Extract ToF statistics (aggregate all sensors)
            tof_raw = tof_data.transpose(2, 3).contiguous().view(B, -1, L)  # [B, 5*64, L]
            tof_stats = self.statistic_extractor(tof_raw, pad_mask)  # [B, 5*64*10]
            tof_stats_processed = self.tof_stats_processor(tof_stats)  # [B, 64]
            
            # Extract Thermal statistics (aggregate all sensors)
            thm_raw = thm_data.transpose(2, 3).contiguous().view(B, -1, L)  # [B, 5*1, L]
            thm_stats = self.statistic_extractor(thm_raw, pad_mask)  # [B, 5*1*10]
            thm_stats_processed = self.thm_stats_processor(thm_stats)  # [B, 64]
        
        # Process sensor data with stats
        imu_features = self.process_sensor_data(self.imu_processor, imu_data, imu_stats_processed, pad_mask)
        tof_features = self.process_sensor_data(self.tof_processor, tof_data, tof_stats_processed, pad_mask)
        thm_features = self.process_sensor_data(self.thm_processor, thm_data, thm_stats_processed, pad_mask)
        
        # Cross attention fusion
        padding_mask = None if pad_mask is None else ~pad_mask.bool()
        fused_features = self.cross_attention_fusion(
            imu_features, tof_features, thm_features, mask=padding_mask
        )  # [B, L, mDim]
        
        # Compute global token
        if pad_mask is not None:
            valid_counts = pad_mask.sum(dim=1, keepdim=True).float()
            valid_counts = torch.clamp(valid_counts, min=1.0)
            global_token = (fused_features * pad_mask.unsqueeze(-1).float()).sum(dim=1) / valid_counts
        else:
            global_token = fused_features.mean(dim=1)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, mDim]
        x = torch.cat((cls_tokens, fused_features), dim=1)  # [B, L+1, mDim]
        
        if pad_mask is not None:
            extended_mask = torch.cat([
                torch.ones(B, 1, device=pad_mask.device, dtype=torch.bool),
                pad_mask
            ], dim=1)  # [B, L+1]
        else:
            extended_mask = None
        
        # Apply transformer layers
        if extended_mask is not None:
            x = self.layer1(x, mask=extended_mask.bool())
            x = self.layer2(x, mask=extended_mask.bool())
        else:
            x = self.layer1(x)
            x = self.layer2(x)
        
        # Extract class token
        cls_token_output = x[:, 0]  # [B, mDim]
        
        # Late fusion with stats (if enabled)
        if cfg.use_stats_vectors:
            imu_stats_late = self.imu_late_fusion_proj(imu_stats_processed)  # [B, mDim]
            tof_stats_late = self.tof_late_fusion_proj(tof_stats_processed)  # [B, mDim]
            thm_stats_late = self.thm_late_fusion_proj(thm_stats_processed)  # [B, mDim]
            
            final_output = torch.cat([
                cls_token_output, 
                (imu_stats_late + tof_stats_late + thm_stats_late) / 3 
            ], dim=1)  # [B, 2*mDim]
        else:
            final_output = cls_token_output  # [B, mDim]
        
        # Apply warmup strategy if needed
        if warmup:
            if cfg.use_stats_vectors:
                # Extend global_token to match fused_output dimension
                global_token_extended = torch.cat([
                    global_token, 
                    torch.zeros_like(global_token),
                    torch.zeros_like(global_token),
                    torch.zeros_like(global_token)
                ], dim=1)  # [B, 4*mDim]
                final_output = 0.1 * final_output + 0.99 * global_token_extended
            else:
                final_output = 0.1 * final_output + 0.99 * global_token
        
        # Final classification
        logits = self._fc2(final_output)
        logits_aux2 = self._fc2_aux2(final_output)
        
        return logits, logits_aux2