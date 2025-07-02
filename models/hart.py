import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# TODO: test it! (iwkms)

# https://github.com/leoinn/hart/blob/main/model.py <-- I HATE TENSORFLOW !!!!!!

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, training=True):
        if not training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class GatedLinearUnit(nn.Module):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.units = units
        self.linear = nn.Linear(units, units * 2)
        
    def forward(self, x):
        projection = self.linear(x)
        linear_proj = projection[..., :self.units]
        gate_proj = torch.sigmoid(projection[..., self.units:])
        return linear_proj * gate_proj

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = nn.Embedding(num_patches, projection_dim)
        
    def forward(self, patch):
        positions = torch.arange(0, self.num_patches, device=patch.device)
        encoded = patch + self.position_embedding(positions)
        return encoded

class ClassToken(nn.Module):
    def __init__(self, hidden_size):
        super(ClassToken, self).__init__()
        self.hidden_size = hidden_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, x], dim=1)

class SensorWiseMHA(nn.Module):
    def __init__(self, projection_quarter, num_heads, start_index, stop_index, 
                 dropout_rate=0.0, drop_path_rate=0.0):
        super(SensorWiseMHA, self).__init__()
        self.projection_quarter = projection_quarter
        self.num_heads = num_heads
        self.start_index = start_index
        self.stop_index = stop_index
        self.mha = nn.MultiheadAttention(
            embed_dim=projection_quarter, 
            num_heads=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.drop_path = DropPath(drop_path_rate)
        
    def forward(self, x, training=True):
        extracted_input = x[:, :, self.start_index:self.stop_index]
        attn_output, _ = self.mha(extracted_input, extracted_input, extracted_input)
        attn_output = self.drop_path(attn_output, training)
        return attn_output

class LiteFormer(nn.Module):
    def __init__(self, start_index, stop_index, projection_size, kernel_size=16, 
                 attention_head=3, drop_path_rate=0.0):
        super(LiteFormer, self).__init__()
        self.start_index = start_index
        self.stop_index = stop_index
        self.kernel_size = kernel_size
        self.projection_size = projection_size
        self.attention_head = attention_head
        self.drop_path = DropPath(drop_path_rate)
        
        self.conv_kernels = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, kernel_size)) 
            for _ in range(attention_head)
        ])
        
    def forward(self, x, training=True):
        # x shape: (batch, seq_len, projection_dim)
        formatted_inputs = x[:, :, self.start_index:self.stop_index]
        batch_size, seq_len, features = formatted_inputs.shape
        
        assert features % self.attention_head == 0, f"Features {features} must be divisible by attention_head {self.attention_head}"
        
        features_per_head = features // self.attention_head
        
        # (batch, seq_len, features) -> (batch * features_per_head, attention_head, seq_len)
        reshaped_inputs = formatted_inputs.reshape(batch_size, seq_len, self.attention_head, features_per_head)
        reshaped_inputs = reshaped_inputs.permute(0, 3, 2, 1)  # (batch, features_per_head, attention_head, seq_len)
        reshaped_inputs = reshaped_inputs.reshape(batch_size * features_per_head, self.attention_head, seq_len)
        
        if training:
            for i, kernel in enumerate(self.conv_kernels):
                self.conv_kernels[i].data = F.softmax(kernel, dim=-1)
        
        conv_outputs = []
        for i in range(self.attention_head):
            conv_out = F.conv1d(
                reshaped_inputs[:, i:i+1, :], 
                self.conv_kernels[i], 
                padding='same'
            )
            conv_outputs.append(conv_out)
        
        conv_outputs = torch.cat(conv_outputs, dim=1)  # (batch * features_per_head, attention_head, seq_len)
        conv_outputs = self.drop_path(conv_outputs, training)
        
        conv_outputs = conv_outputs.reshape(batch_size, features_per_head, self.attention_head, seq_len)
        conv_outputs = conv_outputs.permute(0, 3, 2, 1)  # (batch, seq_len, attention_head, features_per_head)
        local_attention = conv_outputs.reshape(batch_size, seq_len, self.projection_size)
        
        return local_attention

class SensorPatches(nn.Module):
    def __init__(self, projection_dim, patch_size, time_step, num_acc_channels=3, num_gyro_channels=4):
        super(SensorPatches, self).__init__()
        self.patch_size = patch_size
        self.time_step = time_step
        self.projection_dim = projection_dim
        self.num_acc_channels = num_acc_channels
        self.num_gyro_channels = num_gyro_channels
        
        self.acc_projection = nn.Conv1d(
            num_acc_channels, projection_dim // 2, 
            kernel_size=patch_size, stride=time_step
        )
        self.gyro_projection = nn.Conv1d(
            num_gyro_channels, projection_dim // 2, 
            kernel_size=patch_size, stride=time_step
        )
        
    def forward(self, x):
        # x shape: (batch, 1, num_channels, seq_len)
        x = x.squeeze(1)  # (batch, num_channels, seq_len)
        
        acc_data = x[:, :self.num_acc_channels, :]
        gyro_data = x[:, self.num_acc_channels:self.num_acc_channels+self.num_gyro_channels, :]
        
        acc_proj = self.acc_projection(acc_data)  # (batch, proj_dim//2, num_patches)
        gyro_proj = self.gyro_projection(gyro_data)  # (batch, proj_dim//2, num_patches)
        
        projections = torch.cat([acc_proj, gyro_proj], dim=1)  # (batch, proj_dim, num_patches)
        projections = projections.transpose(1, 2)  # (batch, num_patches, proj_dim)
        
        return projections

class HART(nn.Module):
    def __init__(self, input_shape, activity_count, projection_dim=192, 
                 patch_size=16, time_step=16, num_heads=3, filter_attention_head=4,
                 conv_kernels=[3, 7, 15, 31, 31, 31], mlp_head_units=[1024],
                 dropout_rate=0.3, use_tokens=False, num_acc_channels=3, num_gyro_channels=4):
        super(HART, self).__init__()
        
        self.projection_dim = projection_dim
        self.projection_half = projection_dim // 2
        self.projection_quarter = projection_dim // 4
        self.use_tokens = use_tokens
        self.dropout_rate = dropout_rate
        
        self.drop_path_rates = np.linspace(0, dropout_rate * 10, len(conv_kernels)) * 0.1
        
        self.patches = SensorPatches(projection_dim, patch_size, time_step, 
                                   num_acc_channels, num_gyro_channels)
        
        seq_len = input_shape[-1]
        num_patches = (seq_len - patch_size) // time_step + 1
        
        if use_tokens:
            self.class_token = ClassToken(projection_dim)
            num_patches += 1
            
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        
        self.transformer_layers = nn.ModuleList()
        
        for layer_idx, kernel_length in enumerate(conv_kernels):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(projection_dim, eps=1e-6),
                'lite_former': LiteFormer(
                    start_index=self.projection_quarter,
                    stop_index=self.projection_quarter + self.projection_half,
                    projection_size=self.projection_half,
                    attention_head=filter_attention_head,
                    kernel_size=kernel_length,
                    drop_path_rate=self.drop_path_rates[layer_idx]
                ),
                'acc_mha': SensorWiseMHA(
                    self.projection_quarter, num_heads, 0, self.projection_quarter,
                    dropout_rate=dropout_rate, 
                    drop_path_rate=self.drop_path_rates[layer_idx]
                ),
                'gyro_mha': SensorWiseMHA(
                    self.projection_quarter, num_heads, 
                    self.projection_quarter + self.projection_half, projection_dim,
                    dropout_rate=dropout_rate,
                    drop_path_rate=self.drop_path_rates[layer_idx]
                ),
                'norm2': nn.LayerNorm(projection_dim, eps=1e-6),
                'mlp': nn.Sequential(
                    nn.Linear(projection_dim, projection_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(projection_dim * 2, projection_dim)
                ),
                'drop_path': DropPath(self.drop_path_rates[layer_idx])
            })
            self.transformer_layers.append(layer)
        
        self.final_norm = nn.LayerNorm(projection_dim, eps=1e-6)
        
        self.mlp_head = nn.Sequential()
        input_dim = projection_dim
        for units in mlp_head_units:
            self.mlp_head.append(nn.Linear(input_dim, units))
            self.mlp_head.append(nn.SiLU())
            self.mlp_head.append(nn.Dropout(dropout_rate))
            input_dim = units
            
        self.classifier = nn.Linear(input_dim, activity_count)
        
    def forward(self, x):
        # x shape: (batch, 1, num_channels, seq_len)
        
        # Extract patches
        patches = self.patches(x)  # (batch, num_patches, projection_dim)
        
        # Add class token if used
        if self.use_tokens:
            patches = self.class_token(patches)
            
        # Add positional encoding
        encoded_patches = self.patch_encoder(patches)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            # Layer norm
            x1 = layer['norm1'](encoded_patches)
            
            # Multi-branch attention
            branch1 = layer['lite_former'](x1, self.training)
            branch2_acc = layer['acc_mha'](x1, self.training)
            branch2_gyro = layer['gyro_mha'](x1, self.training)
            
            # Concatenate attention outputs
            concat_attention = torch.cat([branch2_acc, branch1, branch2_gyro], dim=2)
            
            # Skip connection 1
            x2 = concat_attention + encoded_patches
            
            # MLP block
            x3 = layer['norm2'](x2)
            x3 = layer['mlp'](x3)
            x3 = layer['drop_path'](x3, self.training)
            
            # Skip connection 2
            encoded_patches = x3 + x2
        
        # Final processing
        representation = self.final_norm(encoded_patches)
        
        if self.use_tokens:
            # Use class token
            representation = representation[:, 0]  # (batch, projection_dim)
        else:
            # Global average pooling
            representation = representation.mean(dim=1)  # (batch, projection_dim)
        
        # Classification head
        features = self.mlp_head(representation)
        logits = self.classifier(features)
        output = F.softmax(logits, dim=-1)
        
        return output

# def create_hart_model(input_shape, num_classes, num_acc_channels=3, num_gyro_channels=4):
#     model = HART(
#         input_shape=input_shape,
#         activity_count=num_classes,
#         projection_dim=192,
#         patch_size=16,
#         time_step=16,
#         num_heads=3,
#         filter_attention_head=4,
#         conv_kernels=[3, 7, 15, 31, 31, 31],
#         mlp_head_units=[1024],
#         dropout_rate=0.3,
#         use_tokens=False,
#         num_acc_channels=num_acc_channels,
#         num_gyro_channels=num_gyro_channels
#     )
#     return model

# if __name__ == "__main__":
#     seq_len = 128
#     num_channels = 16
#     num_classes = 6
    
#     input_shape = (1, num_channels, seq_len)
    
#     model = create_hart_model(
#         input_shape=input_shape,
#         num_classes=num_classes,
#         num_acc_channels=3,  # acc_x, acc_y, acc_z
#         num_gyro_channels=4   # rot_w, rot_x, rot_y, rot_z
#     )
    
#     batch_size = 32
#     test_input = torch.randn(batch_size, 1, num_channels, seq_len)
    
#     model.eval()
#     with torch.no_grad():
#         output = model(test_input)
#         print(f"{test_input.shape}=")
#         print(f"{output.shape}=")