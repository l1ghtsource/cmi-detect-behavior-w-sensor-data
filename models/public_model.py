import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import cfg

# good in hybrid, solo ok

# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=8):
#         super().__init__()
#         self.squeeze = nn.AdaptiveAvgPool1d(1)
#         self.excitation = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         b, c, _ = x.size()
#         y = self.squeeze(x).view(b, c)
#         y = self.excitation(y).view(b, c, 1)
#         return x * y.expand_as(x)

# class ResidualSECNNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3):
#         super().__init__()
        
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
        
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
        
#         self.se = SEBlock(out_channels)
        
#         self.shortcut = nn.Sequential()
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, 1, bias=False),
#                 nn.BatchNorm1d(out_channels)
#             )
        
#         self.pool = nn.MaxPool1d(pool_size)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         shortcut = self.shortcut(x)
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out = self.se(out)
#         out += shortcut
#         out = F.relu(out)
#         out = self.pool(out)
#         out = self.dropout(out)
        
#         return out

# class AttentionLayer(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.attention = nn.Linear(hidden_dim, 1)
        
#     def forward(self, x):
#         # x shape: (batch, seq_len, hidden_dim)
#         scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
#         weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
#         context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
#         return context

# class Public_SingleSensor_v1(nn.Module):
#     def __init__(self, 
#                  channel_size=cfg.imu_vars, 
#                  emb_size=128, 
#                  dim_ff=256, 
#                  dropout=0.3, 
#                  num_classes=cfg.main_num_classes):
#         super().__init__()

#         self.imu_block1 = ResidualSECNNBlock(channel_size, emb_size // 2, 3, dropout=dropout)
#         self.imu_block2 = ResidualSECNNBlock(emb_size // 2, emb_size, 5, dropout=dropout)
        
#         self.bigru = nn.GRU(emb_size, emb_size, bidirectional=True, batch_first=True)
#         self.gru_dropout = nn.Dropout(dropout + 0.1)
        
#         self.attention = AttentionLayer(emb_size * 2)  # 128*2 for bidirectional
        
#         self.dense1 = nn.Linear(emb_size * 2, dim_ff, bias=False)
#         self.bn_dense1 = nn.BatchNorm1d(dim_ff)
#         self.drop1 = nn.Dropout(dropout + 0.2)
        
#         self.dense2 = nn.Linear(dim_ff, dim_ff // 2, bias=False)
#         self.bn_dense2 = nn.BatchNorm1d(dim_ff // 2)
#         self.drop2 = nn.Dropout(dropout)
        
#         self.classifier1 = nn.Linear(dim_ff // 2, num_classes)
#         self.classifier2 = nn.Linear(dim_ff // 2, 2)
        
#     def forward(self, x, pad_mask=False):
#         # input is (bs, 1, T, C)
#         x = x.permute(0, 1, 3, 2) # (bs, 1, C, T)
#         imu = x.squeeze(1) # (bs, C, T)

#         x1 = self.imu_block1(imu)
#         x1 = self.imu_block2(x1)
        
#         merged = x1.transpose(1, 2) # (batch, seq_len, 128)
        
#         gru_out, _ = self.bigru(merged)
#         gru_out = self.gru_dropout(gru_out)
        
#         attended = self.attention(gru_out)
        
#         x = F.relu(self.bn_dense1(self.dense1(attended)))
#         x = self.drop1(x)
#         x = F.relu(self.bn_dense2(self.dense2(x)))
#         x = self.drop2(x)
        
#         logits1 = self.classifier1(x)
#         logits2 = self.classifier2(x)

#         return logits1, logits2

class EnhancedSEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, ks, padding=ks//2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = EnhancedSEBlock(out_channels, reduction=8)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += shortcut
        out = F.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out

class MetaFeatureExtractor(nn.Module):
    def forward(self, x):
        # x shape: (B, L, C)
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        max_val, _ = torch.max(x, dim=1)
        min_val, _ = torch.min(x, dim=1)
        
        # Calculate slope: (last - first) / seq_len
        seq_len = x.size(1)
        if seq_len > 1:
            slope = (x[:, -1, :] - x[:, 0, :]) / (seq_len - 1)
        else:
            slope = torch.zeros_like(x[:, 0, :])
        
        return torch.cat([mean, std, max_val, min_val, slope], dim=1)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))
        weights = F.softmax(scores.squeeze(-1), dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context

class Public_SingleSensor_v1(nn.Module):
    def __init__(
        self,
        channel_size=cfg.imu_vars, 
        num_classes=cfg.main_num_classes
    ):
        super().__init__()

        self.meta_extractor = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * channel_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7]),
                    ResidualSEBlock(36, 48, 3, dropout=0.3),
                    ResidualSEBlock(48, 48, 3, dropout=0.3),
                )
                for _ in range(channel_size)
            ]
        )

        self.bigru = nn.GRU(
            input_size=48 * channel_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.attention_pooling = AttentionLayer(256)

        self.head_1 = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.head_2 = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x, pad_mask=None):
        x = x.squeeze(1) # (bs, l, c)

        meta = self.meta_extractor(x)
        meta_proj = self.meta_dense(meta)

        branch_outputs = []
        for i in range(x.shape[2]):
            channel_input = x[:, :, i].unsqueeze(1)
            processed = self.branches[i](channel_input)
            branch_outputs.append(processed.transpose(1, 2))

        combined = torch.cat(branch_outputs, dim=2)

        gru_out, _ = self.bigru(combined)

        pooled_output = self.attention_pooling(gru_out)

        fused = torch.cat([pooled_output, meta_proj], dim=1)

        z1 = self.head_1(fused)
        z2 = self.head_2(fused)

        return z1, z2