import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_relpos import VisionTransformerRelPos

# idea from https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/discussion/417717

# TODO: test it

class GroupNorm1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.GroupNorm(*args, **kwargs)
        
    def forward(self, x):
        assert len(x.shape) in [2, 3]
        x = x.permute(0, 2, 1) if len(x.shape) == 3 else x
        x = self.norm(x)
        x = x.permute(0, 2, 1) if len(x.shape) == 3 else x
        return x

def getPositionalEncoding(seq_len, d_model, n=10000):
    pos = torch.arange(0, seq_len).unsqueeze(1)
    i = torch.arange(0, d_model, 2).unsqueeze(0)
    enc = torch.zeros(seq_len, d_model)
    enc[:, 0::2] = torch.sin(pos / n ** (i / d_model))
    enc[:, 1::2] = torch.cos(pos / n ** (i / d_model))
    return enc

def alibi_bias(b, a=1):
    b = torch.zeros_like(b)
    n = b.shape[0]//2 + 1
    for h in range(min(8, b.shape[1])):
        bias = -1/2 ** (h + a + 1) * torch.arange(0, n)
        b[:n, h] = torch.flip(bias, [0])
        b[n-1:, h] = bias
    return b

class SimpleLinear(nn.Module):
    def __init__(self, dims, out_dims=None, dropout=0.2, n_layers=1):
        super().__init__()
        self.path = nn.Sequential()
        out_dims = out_dims or dims
        if n_layers == 0: 
            self.path.append(nn.Identity())
        for i in range(n_layers):                
            self.path.append(nn.Sequential(
                nn.Linear(dims if i == 0 else out_dims, out_dims, bias=False),
                nn.LayerNorm(out_dims),
                nn.PReLU(),
                nn.Dropout(dropout),
            ))
    
    def forward(self, x):
        return self.path(x)

class IMU_Backbone(nn.Module):
    def __init__(self, 
                 seq_len,
                 input_channels=7,
                 dims=256, 
                 nheads=12,
                 dropout=0.2,
                 xformer_layers=2,
                 rnn_layers=1,
                 rnn='GRU',
                 xformer_init_1=1.0,
                 xformer_init_2=1.0,
                 xformer_init_scale=0.7,
                 xformer_attn_drop_rate=0.1,
                 xformer_drop_path_rate=0.1,
                 rel_pos=True,
                 alibi=True,
                 pre_norm=False,
                 h0=False):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.dims = dims
        self.pre_norm = pre_norm
        
        self.embed = nn.Sequential(
            nn.Linear(input_channels, dims),
            GroupNorm1d(4, dims),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        
        if xformer_layers > 0:
            class IdentityPatch(nn.Module):
                def __init__(self, **kwargs):
                    super().__init__()
                    self.num_patches = seq_len
                    
                def forward(self, x):
                    return x
                    
            self.xformer = (VisionTransformerRelPos if rel_pos else VisionTransformer)(
                img_size=(seq_len, 1),
                patch_size=(9, 9),
                in_chans=dims,
                num_classes=0,
                global_pool='',
                embed_dim=dims,
                num_heads=nheads,
                embed_layer=IdentityPatch,
                act_layer=nn.GELU,
                depth=xformer_layers,
                init_values=xformer_init_1,
                class_token=False,
                drop_rate=dropout,
                attn_drop_rate=xformer_attn_drop_rate,
                drop_path_rate=xformer_drop_path_rate,
                **({'rel_pos_type': 'bias'} if rel_pos else {})
            )
            
            if rel_pos and alibi:
                for i, b in enumerate(self.xformer.blocks):
                    if hasattr(b.attn, 'rel_pos') and hasattr(b.attn.rel_pos, 'relative_position_bias_table'):
                        b.attn.rel_pos.relative_position_bias_table.data = alibi_bias(
                            b.attn.rel_pos.relative_position_bias_table.data, alibi)
            
            if not rel_pos and hasattr(self.xformer, 'pos_embed'):
                self.xformer.pos_embed.data /= 2
                self.xformer.pos_embed.data[:] += 0.02 * getPositionalEncoding(seq_len, dims, 1000).unsqueeze(0)
            
            for i, b in enumerate(self.xformer.blocks):
                if hasattr(b, 'ls1') and hasattr(b, 'ls2'):
                    b.ls1.gamma.data[:] = torch.tensor(xformer_init_1 * xformer_init_scale ** i)
                    b.ls2.gamma.data[:] = torch.tensor(xformer_init_2 * xformer_init_scale ** i)
        else:
            self.xformer = None
            
        if rnn and rnn_layers > 0:
            self.rnn = getattr(nn, rnn)(
                dims, 
                dims//2,
                num_layers=rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0,
                bidirectional=True,
                batch_first=True
            )
            self.h0 = nn.Parameter(h0 * torch.randn(2 * rnn_layers, dims)) if h0 else None
        else:
            self.rnn = None
            self.h0 = None
            
        self.norm = nn.LayerNorm(dims)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, 1, L, 7] -> [B, L, 7]
        x = x.squeeze(1)
        attn_mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
        x = self.embed(x)  # [B, L, dims]
        
        if self.pre_norm:
            x = self.norm(x)
            
        xt = x
        if self.xformer is not None:
            x = self.xformer(x)
            
        if self.rnn is not None:
            x = self.dropout(x)
            h0 = self.h0.unsqueeze(1).repeat(1, x.shape[0], 1) if self.h0 is not None else None
            x, _ = self.rnn(x, h0)
            
        return x, xt, attn_mask

class CMI_IMU_WalkNetwork(nn.Module):
    def __init__(self,
                 seq_len,
                 input_channels=7,
                 dims=256,
                 nheads=12,
                 dropout=0.2,
                 final_dropout=0.2,
                 target_classes=18,
                 aux_classes=4,
                 aux2_classes=2,
                 xformer_layers=2,
                 rnn_layers=1,
                 rnn='GRU',
                 se_dims=0,
                 se_dropout=0.25,
                 se_pact=0.0,
                 mae_reconstruction=True,
                 **kwargs):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.target_classes = target_classes
        self.aux_classes = aux_classes
        self.aux2_classes = aux2_classes
        self.mae_reconstruction = mae_reconstruction
        self.se_dims = se_dims
        
        self.backbone = IMU_Backbone(
            seq_len=seq_len,
            input_channels=input_channels,
            dims=dims,
            nheads=nheads,
            dropout=dropout,
            xformer_layers=xformer_layers,
            rnn_layers=rnn_layers,
            rnn=rnn,
            **kwargs
        )
        
        rnn_dims = dims if self.backbone.rnn is None else dims
        
        if se_dims > 0:
            self.se_layer = nn.Sequential(
                nn.Dropout(se_dropout),
                nn.Linear(rnn_dims, se_dims),
                nn.PReLU(init=se_pact) if se_pact > 0 else nn.Identity(),
                nn.Dropout(se_dropout),
                nn.Linear(se_dims, rnn_dims),
                nn.Dropout(se_dropout),
                nn.Sigmoid()
            )
        else:
            self.se_layer = None
            
        self.final_dropout = nn.Dropout(final_dropout)
        
        self.target_head = nn.Sequential(
            SimpleLinear(rnn_dims, dims, dropout=dropout),
            nn.Linear(dims, target_classes)
        )
        
        self.aux_head = nn.Sequential(
            SimpleLinear(rnn_dims, dims//2, dropout=dropout),
            nn.Linear(dims//2, aux_classes)
        )
        
        self.aux2_head = nn.Sequential(
            SimpleLinear(rnn_dims, dims//4, dropout=dropout),
            nn.Linear(dims//4, aux2_classes)
        )
        
        if mae_reconstruction:
            self.ae_head = nn.Sequential(
                SimpleLinear(dims, dims * 2, dropout=final_dropout),
                nn.Linear(dims * 2, input_channels)
            )
        else:
            self.ae_head = None
            
        self._init_classification_bias()
        
    def _init_classification_bias(self):
        if hasattr(self.target_head[-1], 'bias') and self.target_head[-1].bias is not None:
            self.target_head[-1].bias.data.fill_(-2.0)
            
        if hasattr(self.aux_head[-1], 'bias') and self.aux_head[-1].bias is not None:
            self.aux_head[-1].bias.data.fill_(-1.5)
            
        if hasattr(self.aux2_head[-1], 'bias') and self.aux2_head[-1].bias is not None:
            self.aux2_head[-1].bias.data.fill_(-1.0)
    
    def forward(self, imu_data):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
        Returns:
            dict with 'target', 'aux', 'aux2' predictions and optionally 'reconstruction'
        """
        x, xt, attn_mask = self.backbone(imu_data)
        
        if self.se_layer is not None:
            se_weights = self.se_layer(x.mean(1))
            
            if self.training:
                se_weights = torch.where(
                    torch.rand_like(se_weights.mean(-1)).unsqueeze(-1) < 0.3,
                    0.5,
                    se_weights
                )
            x = x * se_weights.unsqueeze(1)
        else:
            x = x * 0.5
            
        x_pooled = x.mean(1)  # [B, rnn_dims]
        x_pooled = self.final_dropout(x_pooled)
        
        target_logits = self.target_head(x_pooled)
        aux_logits = self.aux_head(x_pooled)
        aux2_logits = self.aux2_head(x_pooled)
        
        if self.ae_head is not None:
            reconstruction = self.ae_head(xt)
            
        return target_logits, aux_logits, aux2_logits, reconstruction
    
class MultiSensor_Backbone(nn.Module):
    def __init__(self, 
                 seq_len,
                 sensor_configs,
                 dims=256, 
                 nheads=12,
                 dropout=0.2,
                 xformer_layers=2,
                 rnn_layers=1,
                 rnn='GRU',
                 **kwargs):
        super().__init__()
        
        self.seq_len = seq_len
        self.dims = dims
        self.sensor_configs = sensor_configs
        
        self.sensor_backbones = nn.ModuleDict()
        
        for sensor_name, config in sensor_configs.items():
            if sensor_name == 'imu':
                # IMU: [B, 1, L, 7]
                self.sensor_backbones[sensor_name] = self._create_single_sensor_backbone(
                    input_channels=config['channels'],
                    dims=dims,
                    nheads=nheads,
                    dropout=dropout,
                    xformer_layers=xformer_layers,
                    rnn_layers=rnn_layers,
                    rnn=rnn,
                    **kwargs
                )
            else:
                # TOF/THM: [B, num_sensors, L, channels]
                self.sensor_backbones[sensor_name] = self._create_multi_sensor_backbone(
                    num_sensors=config['num_sensors'],
                    input_channels=config['channels'],
                    dims=dims,
                    nheads=nheads,
                    dropout=dropout,
                    xformer_layers=xformer_layers,
                    rnn_layers=rnn_layers,
                    rnn=rnn,
                    **kwargs
                )
    
    def _create_single_sensor_backbone(self, input_channels, dims, nheads, dropout, 
                                     xformer_layers, rnn_layers, rnn, **kwargs):
        return IMU_Backbone(
            seq_len=self.seq_len,
            input_channels=input_channels,
            dims=dims,
            nheads=nheads,
            dropout=dropout,
            xformer_layers=xformer_layers,
            rnn_layers=rnn_layers,
            rnn=rnn,
            **kwargs
        )
    
    def _create_multi_sensor_backbone(self, num_sensors, input_channels, dims, nheads, 
                                    dropout, xformer_layers, rnn_layers, rnn, **kwargs):
        return MultiSensorProcessor(
            seq_len=self.seq_len,
            num_sensors=num_sensors,
            input_channels=input_channels,
            dims=dims,
            nheads=nheads,
            dropout=dropout,
            xformer_layers=xformer_layers,
            rnn_layers=rnn_layers,
            rnn=rnn,
            **kwargs
        )
    
    def forward(self, sensor_data):
        """
        Args:
            sensor_data: dict with keys 'imu', 'tof', 'thm'
        Returns:
            dict with processed features for each sensor type
        """
        sensor_features = {}
        sensor_embeddings = {}
        
        for sensor_name, data in sensor_data.items():
            if sensor_name in self.sensor_backbones:
                features, embeddings, mask = self.sensor_backbones[sensor_name](data)
                sensor_features[sensor_name] = features
                sensor_embeddings[sensor_name] = embeddings
                
        return sensor_features, sensor_embeddings

class MultiSensorProcessor(nn.Module):
    def __init__(self, seq_len, num_sensors, input_channels, dims, nheads, dropout, 
                 xformer_layers, rnn_layers, rnn, **kwargs):
        super().__init__()
        
        self.seq_len = seq_len
        self.num_sensors = num_sensors
        self.input_channels = input_channels
        self.dims = dims
        
        self.embed = nn.Sequential(
            nn.Linear(input_channels, dims),
            GroupNorm1d(4, dims),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        
        if num_sensors > 1:
            self.intra_sensor_attention = nn.MultiheadAttention(
                embed_dim=dims,
                num_heads=nheads//2,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.intra_sensor_attention = None
            
        if xformer_layers > 0:
            class IdentityPatch(nn.Module):
                def __init__(self, **kwargs):
                    super().__init__()
                    self.num_patches = seq_len
                    
                def forward(self, x):
                    return x
                    
            self.xformer = (VisionTransformerRelPos if kwargs.get('rel_pos', True) else VisionTransformer)(
                img_size=(seq_len, 1),
                patch_size=(1, 1),
                in_chans=dims,
                num_classes=0,
                global_pool='',
                embed_dim=dims,
                num_heads=nheads,
                embed_layer=IdentityPatch,
                act_layer=nn.GELU,
                depth=xformer_layers,
                class_token=False,
                drop_rate=dropout,
                **({'rel_pos_type': 'bias'} if kwargs.get('rel_pos', True) else {})
            )
        else:
            self.xformer = None
            
        if rnn and rnn_layers > 0:
            self.rnn = getattr(nn, rnn)(
                dims, 
                dims//2,
                num_layers=rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0,
                bidirectional=True,
                batch_first=True
            )
        else:
            self.rnn = None
            
        self.norm = nn.LayerNorm(dims)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [B, num_sensors, L, channels]
        Returns:
            features, embeddings, mask
        """
        B, num_sensors, L, C = x.shape
        x_reshaped = x.view(B * num_sensors, L, C)
        x_embedded = self.embed(x_reshaped)  # [B*num_sensors, L, dims]
        
        xt = x_embedded
        if self.xformer is not None:
            x_embedded = self.xformer(x_embedded)
            
        if self.rnn is not None:
            x_embedded = self.dropout(x_embedded)
            x_embedded, _ = self.rnn(x_embedded)
        
        x_embedded = x_embedded.view(B, num_sensors, L, self.dims)
        
        if self.intra_sensor_attention is not None and num_sensors > 1:
            attended_features = []
            for t in range(L):
                sensor_features_t = x_embedded[:, :, t, :]  # [B, num_sensors, dims]
                attended_t, _ = self.intra_sensor_attention(
                    sensor_features_t, sensor_features_t, sensor_features_t
                )
                attended_features.append(attended_t)
            x_embedded = torch.stack(attended_features, dim=2)  # [B, num_sensors, L, dims]
        
        x_pooled = x_embedded.mean(1)  # [B, L, dims] - усредняем по сенсорам
        
        attn_mask = torch.ones(B, L, device=x.device)
        
        return x_pooled, xt.view(B, num_sensors, L, self.dims).mean(1), attn_mask

class CrossSensorAttention(nn.Module):
    def __init__(self, dims, nheads, dropout=0.1):
        super().__init__()
        
        self.dims = dims
        self.nheads = nheads
        
        self.cross_attention = nn.ModuleDict()
        sensor_pairs = [
            ('imu', 'tof'), ('imu', 'thm'), ('tof', 'thm'),
            ('tof', 'imu'), ('thm', 'imu'), ('thm', 'tof')
        ]
        
        for query_sensor, key_sensor in sensor_pairs:
            self.cross_attention[f"{query_sensor}_{key_sensor}"] = nn.MultiheadAttention(
                embed_dim=dims,
                num_heads=nheads,
                dropout=dropout,
                batch_first=True
            )
        
        self.norm = nn.LayerNorm(dims)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sensor_features):
        """
        Args:
            sensor_features: dict with 'imu', 'tof', 'thm' features [B, L, dims]
        Returns:
            enhanced_features: dict with cross-attended features
        """
        enhanced_features = {}
        
        for sensor_name, features in sensor_features.items():
            enhanced_feature = features
            
            for other_sensor, other_features in sensor_features.items():
                if other_sensor != sensor_name:
                    attention_key = f"{sensor_name}_{other_sensor}"
                    if attention_key in self.cross_attention:
                        attended, _ = self.cross_attention[attention_key](
                            features, other_features, other_features
                        )
                        enhanced_feature = enhanced_feature + self.dropout(attended)
            
            enhanced_features[sensor_name] = self.norm(enhanced_feature)
            
        return enhanced_features

class CMI_MultiSensor_WalkNetwork(nn.Module):
    def __init__(self,
                 seq_len,
                 dims=256,
                 nheads=12,
                 dropout=0.2,
                 final_dropout=0.2,
                 target_classes=18,
                 aux_classes=4,
                 aux2_classes=2,
                 xformer_layers=2,
                 rnn_layers=1,
                 rnn='GRU',
                 se_dims=0,
                 se_dropout=0.25,
                 cross_attention=True,
                 mae_reconstruction=True,
                 **kwargs):
        super().__init__()
        
        self.seq_len = seq_len
        self.target_classes = target_classes
        self.aux_classes = aux_classes
        self.aux2_classes = aux2_classes
        self.cross_attention_enabled = cross_attention
        self.mae_reconstruction = mae_reconstruction
        
        sensor_configs = {
            'imu': {'num_sensors': 1, 'channels': 7},
            'tof': {'num_sensors': 5, 'channels': 64},
            'thm': {'num_sensors': 5, 'channels': 1}
        }
        
        self.backbone = MultiSensor_Backbone(
            seq_len=seq_len,
            sensor_configs=sensor_configs,
            dims=dims,
            nheads=nheads,
            dropout=dropout,
            xformer_layers=xformer_layers,
            rnn_layers=rnn_layers,
            rnn=rnn,
            **kwargs
        )
        
        if cross_attention:
            self.cross_attention = CrossSensorAttention(dims, nheads//2, dropout)
        else:
            self.cross_attention = None
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(dims * 3, dims * 2),
            nn.LayerNorm(dims * 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(dims * 2, dims),
            nn.LayerNorm(dims),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        
        if se_dims > 0:
            self.se_layer = nn.Sequential(
                nn.Dropout(se_dropout),
                nn.Linear(dims, se_dims),
                nn.PReLU(),
                nn.Dropout(se_dropout),
                nn.Linear(se_dims, dims),
                nn.Dropout(se_dropout),
                nn.Sigmoid()
            )
        else:
            self.se_layer = None
            
        self.final_dropout = nn.Dropout(final_dropout)
        
        self.target_head = nn.Sequential(
            SimpleLinear(dims, dims, dropout=dropout),
            nn.Linear(dims, target_classes)
        )
        
        self.aux_head = nn.Sequential(
            SimpleLinear(dims, dims//2, dropout=dropout),
            nn.Linear(dims//2, aux_classes)
        )
        
        self.aux2_head = nn.Sequential(
            SimpleLinear(dims, dims//4, dropout=dropout),
            nn.Linear(dims//4, aux2_classes)
        )
        
        if mae_reconstruction:
            self.ae_heads = nn.ModuleDict({
                'imu': nn.Sequential(
                    SimpleLinear(dims, dims * 2, dropout=final_dropout),
                    nn.Linear(dims * 2, 7)
                ),
                'tof': nn.Sequential(
                    SimpleLinear(dims, dims * 2, dropout=final_dropout),
                    nn.Linear(dims * 2, 64)
                ),
                'thm': nn.Sequential(
                    SimpleLinear(dims, dims * 2, dropout=final_dropout),
                    nn.Linear(dims * 2, 1)
                )
            })
        else:
            self.ae_heads = None
            
        self._init_classification_bias()
        
    def _init_classification_bias(self):
        if hasattr(self.target_head[-1], 'bias') and self.target_head[-1].bias is not None:
            self.target_head[-1].bias.data.fill_(-2.0)
            
        if hasattr(self.aux_head[-1], 'bias') and self.aux_head[-1].bias is not None:
            self.aux_head[-1].bias.data.fill_(-1.5)
            
        if hasattr(self.aux2_head[-1], 'bias') and self.aux2_head[-1].bias is not None:
            self.aux2_head[-1].bias.data.fill_(-1.0)
    
    def forward(self, imu_data, thm_data, tof_data):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data
            thm_data: [B, 5, L, 1] - Thermal sensor data
        Returns:
            target_logits, aux_logits, aux2_logits, reconstructions
        """
        sensor_data = {
            'imu': imu_data,
            'tof': tof_data,
            'thm': thm_data
        }
        
        sensor_features, sensor_embeddings = self.backbone(sensor_data)
        
        if self.cross_attention is not None:
            sensor_features = self.cross_attention(sensor_features)
        
        combined_features = torch.cat([
            sensor_features['imu'].mean(1),    # [B, dims]
            sensor_features['tof'].mean(1),    # [B, dims]
            sensor_features['thm'].mean(1)     # [B, dims]
        ], dim=-1)  # [B, dims*3]
        
        fused_features = self.fusion_layer(combined_features)  # [B, dims]
        
        if self.se_layer is not None:
            se_weights = self.se_layer(fused_features)
            if self.training:
                se_weights = torch.where(
                    torch.rand_like(se_weights.mean(-1)).unsqueeze(-1) < 0.3,
                    0.5,
                    se_weights
                )
            fused_features = fused_features * se_weights
        else:
            fused_features = fused_features * 0.5
            
        fused_features = self.final_dropout(fused_features)
        
        target_logits = self.target_head(fused_features)
        aux_logits = self.aux_head(fused_features)
        aux2_logits = self.aux2_head(fused_features)
        
        reconstructions = {}
        if self.ae_heads is not None:
            for sensor_name, ae_head in self.ae_heads.items():
                if sensor_name in sensor_embeddings:
                    reconstructions[sensor_name] = ae_head(sensor_embeddings[sensor_name])
        
        return target_logits, aux_logits, aux2_logits, reconstructions