import torch
import torch.nn as nn
import numpy as np
from modules.inceptiontime import InceptionTimeFeatureExtractor
from modules.nystrom_attention import NystromAttention
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

    def forward(self, x):
        x = x + self.attn(self.norm(x))
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

        # 128 cuz InceptionModule do x4 for out_dim !!
        total_features = (cfg.imu_vars + cfg.tof_vars + cfg.thm_vars) * 128  # 1408 total features
        
        # Projection layer to map concatenated features to target dimension
        self.feature_proj = nn.Linear(total_features, mDim)
            
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
        
        self._fc2 = nn.Sequential(
            nn.Linear(mDim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        ) 
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        initialize_weights(self)
        
    def forward(self, imu_data, thm_data, tof_data, warmup=False):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data  
            thm_data: [B, 5, L, 1] - Thermal sensor data
            warmup: bool - whether to use warmup strategy
        """
        B, _, L, _ = imu_data.shape
        
        # Process each sensor type and collect features
        all_features = []
        
        # Process IMU data (1 sensor)
        for i in range(cfg.imu_num_sensor):
            sensor_data = imu_data[:, i, :, :].transpose(1, 2)  # [B, 7, L]
            features = self.imu_feature_extractor(sensor_data)  # [B, 128, L]
            all_features.append(features)
        
        # Process ToF data (5 sensors)
        for i in range(cfg.tof_num_sensor):
            sensor_data = tof_data[:, i, :, :].transpose(1, 2)  # [B, 64, L]
            features = self.tof_feature_extractor(sensor_data)  # [B, 128, L]
            all_features.append(features)
        
        # Process Thermal data (5 sensors)
        for i in range(cfg.thm_num_sensor):
            sensor_data = thm_data[:, i, :, :].transpose(1, 2)  # [B, 1, L]
            features = self.thm_feature_extractor(sensor_data)  # [B, 128, L]
            all_features.append(features)
        
        # Concatenate all features along channel dimension
        x = torch.cat(all_features, dim=1)  # [B, 11*128, L]
        
        # Transpose and project to target dimension
        x = x.transpose(1, 2)  # [B, L, 11*128]
        x = self.feature_proj(x)  # [B, L, mDim]

        # Compute global token
        global_token = x.mean(dim=1)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, mDim]
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Wavelet Positional Encoding 1
        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3)
        
        # Apply TransLayer 1
        x = self.layer1(x)
        
        # Apply Wavelet Positional Encoding 2
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_)
        
        # Apply TransLayer 2
        x = self.layer2(x)
        
        # Extract class token
        x = x[:, 0]
        
        # Apply warmup strategy if needed
        if warmup:
            x = 0.1 * x + 0.99 * global_token   
 
        # Final classification
        logits = self._fc2(x)
            
        return logits

# ya ebal eto govnishe
class TimeMIL_SingleSensor_v1(nn.Module):
    def __init__(self, n_classes=cfg.num_classes, mDim=cfg.timemil_dim, max_seq_len=cfg.seq_len, dropout=cfg.timemil_dropout):
        super().__init__()
     
        # Define feature extractor for IMU sensor only
        self.imu_feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=cfg.imu_vars)

        # 128 cuz InceptionModule do x4 for out_dim !!
        # Only 1 IMU sensor with cfg.imu_vars channels -> 128 features
        total_features = 128  # 1 sensor * 128 features
        
        # Projection layer to map features to target dimension
        self.feature_proj = nn.Linear(total_features, mDim)
            
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
        
        self._fc2 = nn.Sequential(
            nn.Linear(mDim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        ) 
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        initialize_weights(self)
        
    def forward(self, imu_data, warmup=False):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            warmup: bool - whether to use warmup strategy
        """
        B, _, L, _ = imu_data.shape
        
        # Process IMU data (single sensor)
        sensor_data = imu_data[:, 0, :, :].transpose(1, 2)  # [B, 7, L]
        x = self.imu_feature_extractor(sensor_data)  # [B, 128, L]
        
        # Transpose and project to target dimension
        x = x.transpose(1, 2)  # [B, L, 128]
        x = self.feature_proj(x)  # [B, L, mDim]

        # Compute global token
        global_token = x.mean(dim=1)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, mDim]
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Wavelet Positional Encoding 1
        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3)
        
        # Apply TransLayer 1
        x = self.layer1(x)
        
        # Apply Wavelet Positional Encoding 2
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_)
        
        # Apply TransLayer 2
        x = self.layer2(x)
        
        # Extract class token
        x = x[:, 0]
        
        # Apply warmup strategy if needed
        if warmup:
            x = 0.1 * x + 0.99 * global_token   
 
        # Final classification
        logits = self._fc2(x)
            
        return logits