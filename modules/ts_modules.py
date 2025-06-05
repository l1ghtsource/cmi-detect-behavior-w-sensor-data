import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=3, 
            dilation=dilation_rate,
            padding='same'
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class SensorAttn(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        return F.scaled_dot_product_attention(Q, K, V)
    
class DemographicsEncoder(nn.Module):
    def __init__(self, demo_features=7, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(demo_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# class HierarchicalDemographicsEncoder(nn.Module):
#     def __init__(self, demo_features=7, hidden_dim=64, dropout=0.2):
#         super().__init__()
        
#         self.basic_demo_encoder = nn.Sequential(
#             nn.Linear(3, hidden_dim // 4),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim // 4),
#             nn.Dropout(dropout)
#         )
        
#         self.physical_encoder = nn.Sequential(
#             nn.Linear(4, hidden_dim // 4),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim // 4),
#             nn.Dropout(dropout)
#         )
        
#         self.cross_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim // 4,
#             num_heads=2,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         self.hierarchical_fusion = nn.Sequential(
#             nn.Linear(hidden_dim // 2, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.GLU(dim=1),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
        
#     def forward(self, x):
#         # x: [batch_size, 7]
        
#         basic_demo = x[:, [0, 1, 2]]  # adult_child, age, sex
#         physical = x[:, [3, 4, 5, 6]]  # handedness, height_cm, shoulder_to_wrist_cm, elbow_to_wrist_cm
        
#         basic_encoded = self.basic_demo_encoder(basic_demo)  # [batch_size, hidden_dim//4]
#         physical_encoded = self.physical_encoder(physical)   # [batch_size, hidden_dim//4]
        
#         basic_expanded = basic_encoded.unsqueeze(1)    # [batch_size, 1, hidden_dim//4]
#         physical_expanded = physical_encoded.unsqueeze(1)  # [batch_size, 1, hidden_dim//4]
        
#         basic_attended, _ = self.cross_attention(basic_expanded, physical_expanded, physical_expanded)
#         physical_attended, _ = self.cross_attention(physical_expanded, basic_expanded, basic_expanded)
        
#         basic_attended = basic_attended.squeeze(1)
#         physical_attended = physical_attended.squeeze(1)
        
#         combined = torch.cat([basic_attended, physical_attended], dim=1)
        
#         return self.hierarchical_fusion(combined)

# class ResidualDemographicsEncoder(nn.Module):
#     def __init__(self, demo_features=7, hidden_dim=64, dropout=0.2, num_layers=3):
#         super().__init__()
        
#         self.input_projection = nn.Linear(demo_features, hidden_dim)
        
#         self.residual_blocks = nn.ModuleList([
#             self._make_residual_block(hidden_dim, dropout) 
#             for _ in range(num_layers)
#         ])
        
#         self.skip_weights = nn.Parameter(torch.ones(num_layers + 1))
        
#         self.final_norm = nn.LayerNorm(hidden_dim)
        
#     def _make_residual_block(self, hidden_dim, dropout):
#         return nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.Dropout(dropout)
#         )
    
#     def forward(self, x):
#         x = self.input_projection(x)
        
#         representations = [x]
        
#         current = x
#         for block in self.residual_blocks:
#             residual = current
#             current = block(current) + residual  # residual connection
#             representations.append(current)
        
#         weighted_sum = sum(
#             weight * repr for weight, repr in zip(self.skip_weights, representations)
#         )
        
#         return self.final_norm(weighted_sum)