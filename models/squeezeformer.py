import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, mask=None):
        x, gate = torch.split(x, x.size(-1) // 2, dim=-1)
        x = x * F.silu(gate)
        return x

class GLUMlp(nn.Module):
    def __init__(self, dim_expand, dim):
        super().__init__()
        self.dim_expand = dim_expand
        self.dim = dim
        self.dense_1 = nn.Linear(dim, dim_expand, bias=True)
        self.glu_1 = GLU()
        self.dense_2 = nn.Linear(dim_expand // 2, dim, bias=True)
    
    def forward(self, x, training=False):
        x = self.dense_1(x)
        x = self.glu_1(x)
        x = self.dense_2(x)
        return x

class ScaleBias(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x, mask=None):
        return x * self.scale + self.bias

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = GLUMlp(feed_forward_dim, embed_dim)
        self.layer_norm_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.scale_bias_1 = ScaleBias(embed_dim)
        self.scale_bias_2 = ScaleBias(embed_dim)
    
    def forward(self, x, pad_mask=None, training=None):
        residual = x
        
        attn_mask = None
        if pad_mask is not None:
            attn_mask = ~pad_mask.bool()  # [B, L]
        
        x, _ = self.att(x, x, x, key_padding_mask=attn_mask)
        x = self.scale_bias_1(x)
        x = self.layer_norm_1(x + residual)
        
        residual = x
        x = self.ffn(x, training=training)
        x = self.scale_bias_2(x)
        x = self.layer_norm_2(x + residual)
        return x

class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             stride=1, padding=kernel_size//2, bias=False)
    
    def forward(self, inputs):
        # inputs shape: (batch, seq_len, channels)
        nn_out = F.adaptive_avg_pool1d(inputs.transpose(1, 2), 1)  # (batch, channels, 1)
        nn_out = nn_out.transpose(1, 2)  # (batch, 1, channels)
        nn_out = self.conv(nn_out.transpose(1, 2))  # (batch, 1, channels)
        nn_out = torch.sigmoid(nn_out.transpose(1, 2))  # (batch, 1, channels)
        return inputs * nn_out

class HeadDense(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.dense = None
    
    def forward(self, x):
        if self.dense is None:
            self.dense = nn.Linear(x.size(-1), self.head_dim).to(x.device)
        x = F.silu(self.dense(x))
        return x

class Conv1DBlockSqueezeformer(nn.Module):
    def __init__(self, channel_size, kernel_size, dilation_rate=1,
                 expand_ratio=4, se_ratio=0.25, activation='swish'):
        super().__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.activation = activation
        
        self.scale_bias = ScaleBias(channel_size)
        self.glu_layer = GLU()
        self.ffn = GLUMlp(channel_size * 4, channel_size)
        self.layer_norm_2 = nn.LayerNorm(channel_size, eps=1e-6)
        self.scale_bias_1 = ScaleBias(channel_size)
        self.scale_bias_2 = ScaleBias(channel_size)
        
        self.expand = None
        self.project = None
        self.dwconv = None
        self.batch_norm = None
        self.eca_layer = ECA()
        self.layernorm = nn.LayerNorm(channel_size, eps=1e-6)
    
    def forward(self, x, training=None):
        if self.expand is None:
            channels_in = x.size(-1)
            channels_expand = channels_in * self.expand_ratio
            
            self.expand = nn.Linear(channels_in, channels_expand, bias=True).to(x.device)
            self.project = nn.Linear(channels_expand // 2, self.channel_size, bias=True).to(x.device)
            self.dwconv = nn.Conv1d(channels_expand // 2, channels_expand // 2, 
                                   self.kernel_size, dilation=self.dilation_rate,
                                   padding=self.dilation_rate * (self.kernel_size - 1) // 2,
                                   groups=channels_expand // 2, bias=False).to(x.device)
            self.batch_norm = nn.BatchNorm1d(channels_expand // 2, momentum=0.05).to(x.device)
        
        skip = x
        
        x = self.expand(x)
        x = self.glu_layer(x)
        
        #  (batch, seq, channels) -> (batch, channels, seq)
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = self.batch_norm(x)
        x = F.silu(x)  # swish activation
        x = x.transpose(1, 2)  # (batch, seq, channels)
        
        x = self.eca_layer(x)
        x = self.project(x)
        x = self.scale_bias_1(x)
        
        x = x + skip
        
        residual = x
        x = self.ffn(x)
        x = self.scale_bias_2(x)
        x = self.layer_norm_2(x + residual)
        return x

class Squeezeformer_SingleSensor_v1(nn.Module):
    def __init__(self, dim=384, head_dim=2048):
        super().__init__()
        
        self.input_dense = nn.Linear(7, dim, bias=False)
        self.input_norm = nn.LayerNorm(dim, eps=1e-6)
        
        conv_filter = 15
        self.blocks = nn.ModuleList([
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
            Conv1DBlockSqueezeformer(dim, conv_filter),
            TransformerEncoder(dim, 4, dim * 4),
        ])
        
        self.head_dense = HeadDense(head_dim)
        self.head_mlp = GLUMlp(head_dim * 2, head_dim)
        
        self.label1_head = nn.Linear(head_dim, 18)  # 18 
        self.label2_head = nn.Linear(head_dim, 2)   # 2 
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, imu_data, pad_mask=None):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            pad_mask: [B, L] - padding mask (1=valid, 0=padding) [optional!]
        """
        x = imu_data.squeeze(1)
        
        x = self.input_dense(x)
        x = self.input_norm(x)
        
        for block in self.blocks:
            if isinstance(block, TransformerEncoder):
                x = block(x, pad_mask=pad_mask)
            else:
                x = block(x)
        
        x = self.head_dense(x)
        x = self.head_mlp(x)
        x = self.dropout(x)
        
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(-1).float()  # [B, L, 1]
            x_masked = x * mask
            x_pooled = x_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)  # [B, head_dim]
        else:
            x_pooled = x.mean(dim=1)  # [B, head_dim]
        
        label1 = self.label1_head(x_pooled)  # [B, 18]
        label2 = self.label2_head(x_pooled)  # [B, 2]
        
        return label1, label2
    
class CrossSensorAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.scale_bias = ScaleBias(embed_dim)
        
    def forward(self, x, pad_mask=None):
        """
        Args:
            x: [B, num_sensors, L, embed_dim]
            pad_mask: [B, L] - padding mask (1=valid, 0=padding)
        Returns:
            x: [B, num_sensors, L, embed_dim]
        """
        B, num_sensors, L, embed_dim = x.shape
        
        # [B, num_sensors, L, embed_dim] -> [B*L, num_sensors, embed_dim]
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * L, num_sensors, embed_dim)
        
        sensor_mask = None
        if pad_mask is not None:
            # pad_mask: [B, L] -> [B*L] -> [B*L, num_sensors]
            time_mask = pad_mask.view(B * L)  # [B*L]
            sensor_mask = ~time_mask.unsqueeze(1).repeat(1, num_sensors)  # [B*L, num_sensors]
        
        residual = x_reshaped
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped, 
                                    key_padding_mask=sensor_mask)
        attn_out = self.scale_bias(attn_out)
        x_out = self.layer_norm(attn_out + residual)
        
        x_out = x_out.reshape(B, L, num_sensors, embed_dim).permute(0, 2, 1, 3)
        
        return x_out

class SensorEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers=2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, 1000, embed_dim) * 0.02)
        
        self.temporal_blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
    def forward(self, x, pad_mask=None):
        """
        Args:
            x: [B, num_sensors, L, input_dim]
            pad_mask: [B, L] - padding mask (1=valid, 0=padding)
        Returns:
            x: [B, num_sensors, L, embed_dim]
        """
        B, num_sensors, L, input_dim = x.shape
        
        x = self.input_projection(x)  # [B, num_sensors, L, embed_dim]
        x = x + self.pos_encoding[:, :, :L, :]
        
        x_flat = x.reshape(B * num_sensors, L, -1)  # [B*num_sensors, L, embed_dim]
        
        if pad_mask is not None:
            # [B, L] -> [B*num_sensors, L]
            sensor_pad_mask = pad_mask.unsqueeze(1).repeat(1, num_sensors, 1).view(B * num_sensors, L)
        else:
            sensor_pad_mask = None
        
        for block in self.temporal_blocks:
            x_flat = block(x_flat, pad_mask=sensor_pad_mask)
        
        x = x_flat.reshape(B, num_sensors, L, -1)  # [B, num_sensors, L, embed_dim]
        x = self.norm(x)
        
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.cross_attentions = nn.ModuleDict({
            'imu_tof': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
            'imu_thm': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
            'tof_thm': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
        })
        
        self.layer_norms = nn.ModuleDict({
            'self': nn.LayerNorm(embed_dim, eps=1e-6),
            'imu_tof': nn.LayerNorm(embed_dim, eps=1e-6),
            'imu_thm': nn.LayerNorm(embed_dim, eps=1e-6),
            'tof_thm': nn.LayerNorm(embed_dim, eps=1e-6),
        })
        
        self.scale_biases = nn.ModuleDict({
            'self': ScaleBias(embed_dim),
            'imu_tof': ScaleBias(embed_dim),
            'imu_thm': ScaleBias(embed_dim),
            'tof_thm': ScaleBias(embed_dim),
        })
        
        self.fusion_mlp = GLUMlp(embed_dim * 4, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
    def forward(self, imu_features, tof_features, thm_features, pad_mask=None):
        """
        Args:
            imu_features: [B, L, embed_dim] - агрегированные IMU признаки
            tof_features: [B, L, embed_dim] - агрегированные ToF признаки  
            thm_features: [B, L, embed_dim] - агрегированные THM признаки
            pad_mask: [B, L] - padding mask (1=valid, 0=padding)
        """
        B, L, embed_dim = imu_features.shape
        
        key_padding_mask = None
        if pad_mask is not None:
            key_padding_mask = ~pad_mask.bool()  # [B, L]
        
        all_features = torch.stack([imu_features, tof_features, thm_features], dim=1)  # [B, 3, L, embed_dim]
        all_features_flat = all_features.reshape(B, 3 * L, embed_dim)  # [B, 3*L, embed_dim]
        
        if key_padding_mask is not None:
            extended_mask = key_padding_mask.repeat(1, 3)  # [B, 3*L]
        else:
            extended_mask = None
        
        residual = all_features_flat
        self_attn_out, _ = self.self_attention(all_features_flat, all_features_flat, all_features_flat,
                                              key_padding_mask=extended_mask)
        self_attn_out = self.scale_biases['self'](self_attn_out)
        self_attn_out = self.layer_norms['self'](self_attn_out + residual)
        
        self_attn_out = self_attn_out.reshape(B, 3, L, embed_dim)
        imu_self, tof_self, thm_self = self_attn_out[:, 0], self_attn_out[:, 1], self_attn_out[:, 2]
        
        # IMU <-> ToF
        imu_residual = imu_self
        imu_cross_tof, _ = self.cross_attentions['imu_tof'](imu_self, tof_self, tof_self,
                                                           key_padding_mask=key_padding_mask)
        imu_cross_tof = self.scale_biases['imu_tof'](imu_cross_tof)
        imu_enhanced = self.layer_norms['imu_tof'](imu_cross_tof + imu_residual)
        
        # IMU <-> THM
        imu_residual = imu_enhanced
        imu_cross_thm, _ = self.cross_attentions['imu_thm'](imu_enhanced, thm_self, thm_self,
                                                           key_padding_mask=key_padding_mask)
        imu_cross_thm = self.scale_biases['imu_thm'](imu_cross_thm)
        imu_final = self.layer_norms['imu_thm'](imu_cross_thm + imu_residual)
        
        # ToF <-> THM
        tof_residual = tof_self
        tof_cross_thm, _ = self.cross_attentions['tof_thm'](tof_self, thm_self, thm_self,
                                                           key_padding_mask=key_padding_mask)
        tof_cross_thm = self.scale_biases['tof_thm'](tof_cross_thm)
        tof_final = self.layer_norms['tof_thm'](tof_cross_thm + tof_residual)
        
        fused_features = torch.cat([imu_final, tof_final, thm_self, 
                                   (imu_final + tof_final + thm_self) / 3], dim=-1)  # [B, L, 4*embed_dim]
        fused_features = self.fusion_mlp(fused_features)  # [B, L, embed_dim]
        fused_features = self.fusion_norm(fused_features)
        
        return fused_features

class Squeezeformer_MultiSensor_v1(nn.Module):
    def __init__(self, embed_dim=384, num_heads=8, head_dim=2048):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.imu_encoder = SensorEncoder(7, embed_dim, num_heads, num_layers=3)
        self.tof_encoder = SensorEncoder(64, embed_dim, num_heads, num_layers=3)
        self.thm_encoder = SensorEncoder(1, embed_dim, num_heads, num_layers=3)
        
        self.tof_cross_sensor = CrossSensorAttention(embed_dim, num_heads)
        self.thm_cross_sensor = CrossSensorAttention(embed_dim, num_heads)
        
        self.sensor_aggregation = nn.ModuleDict({
            'tof': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
            'thm': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
        })
        
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads)
            for _ in range(2)
        ])
        
        self.final_encoder = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, embed_dim * 4)
            for _ in range(2)
        ])
        
        self.head_dense = nn.Linear(embed_dim, head_dim)
        self.head_mlp = GLUMlp(head_dim * 2, head_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.label1_head = nn.Linear(head_dim, 18)  # 18 классов
        self.label2_head = nn.Linear(head_dim, 2)   # 2 класса
        
    def forward(self, imu_data, thm_data, tof_data, pad_mask=None):
        """
        Args:
            imu_data: [B, 1, L, 7] - IMU sensor data
            tof_data: [B, 5, L, 64] - Time-of-Flight sensor data
            thm_data: [B, 5, L, 1] - Thermal sensor data
            pad_mask: [B, L] - padding mask (1=valid, 0=padding) [optional!]
        """
        B, _, L, _ = imu_data.shape
        
        imu_features = self.imu_encoder(imu_data, pad_mask=pad_mask)    # [B, 1, L, embed_dim]
        tof_features = self.tof_encoder(tof_data, pad_mask=pad_mask)    # [B, 5, L, embed_dim]
        thm_features = self.thm_encoder(thm_data, pad_mask=pad_mask)    # [B, 5, L, embed_dim]
        
        tof_features = self.tof_cross_sensor(tof_features, pad_mask=pad_mask)  # [B, 5, L, embed_dim]
        thm_features = self.thm_cross_sensor(thm_features, pad_mask=pad_mask)  # [B, 5, L, embed_dim]
        
        imu_agg = imu_features.squeeze(1)  # [B, L, embed_dim]
        
        tof_flat = tof_features.permute(0, 2, 1, 3).reshape(B * L, 5, self.embed_dim)  # [B*L, 5, embed_dim]
        
        sensor_mask = None
        if pad_mask is not None:
            time_mask = pad_mask.view(B * L)  # [B*L]
            sensor_mask = ~time_mask.unsqueeze(1).repeat(1, 5)  # [B*L, 5] - True для игнорирования
        
        tof_agg, _ = self.sensor_aggregation['tof'](tof_flat, tof_flat, tof_flat,
                                                   key_padding_mask=sensor_mask)
        tof_agg = tof_agg.mean(dim=1)  # [B*L, embed_dim]
        tof_agg = tof_agg.reshape(B, L, self.embed_dim)  # [B, L, embed_dim]
        
        thm_flat = thm_features.permute(0, 2, 1, 3).reshape(B * L, 5, self.embed_dim)  # [B*L, 5, embed_dim]
        thm_agg, _ = self.sensor_aggregation['thm'](thm_flat, thm_flat, thm_flat,
                                                   key_padding_mask=sensor_mask)
        thm_agg = thm_agg.mean(dim=1)  # [B*L, embed_dim]
        thm_agg = thm_agg.reshape(B, L, self.embed_dim)  # [B, L, embed_dim]
        
        fused_features = None
        for cross_modal_layer in self.cross_modal_layers:
            if fused_features is None:
                fused_features = cross_modal_layer(imu_agg, tof_agg, thm_agg, pad_mask=pad_mask)
            else:
                fused_features = cross_modal_layer(fused_features, fused_features, fused_features, 
                                                 pad_mask=pad_mask)
        
        for encoder in self.final_encoder:
            fused_features = encoder(fused_features, pad_mask=pad_mask)
        
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(-1).float()  # [B, L, 1]
            fused_masked = fused_features * mask
            temporal_features = fused_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)  # [B, embed_dim]
        else:
            temporal_features = fused_features.mean(dim=1)  # [B, embed_dim]
        
        x = F.silu(self.head_dense(temporal_features))
        x = self.head_mlp(x)
        x = self.dropout(x)
        
        label1 = self.label1_head(x)  # [B, 18]
        label2 = self.label2_head(x)  # [B, 2]
        
        return label1, label2