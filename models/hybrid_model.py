import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.inceptiontime_replacers import Resnet1DFeatureExtractor
from models.convtran import SEBlock228
from models.filternet import FilterNet_SingleSensor_v1, DEFAULT_WIDTH
from models.basic_cnn1ds import SEPlusMean, MLPNeck
from models.multi_bigru import ResidualBiGRU
from configs.config import cfg

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = SEBlock(out_channels)
        
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

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context

class Public_SingleSensor_Extractor(nn.Module):
    def __init__(self, 
                 channel_size=cfg.imu_vars, 
                 emb_size=128, 
                 dim_ff=256, 
                 dropout=0.3):
        super().__init__()

        self.imu_block1 = ResidualSECNNBlock(channel_size, emb_size // 2, 3, dropout=dropout)
        self.imu_block2 = ResidualSECNNBlock(emb_size // 2, emb_size, 5, dropout=dropout)
        
        self.bigru = nn.GRU(emb_size, emb_size, bidirectional=True, batch_first=True)
        self.gru_dropout = nn.Dropout(dropout + 0.1)
        
        self.attention = AttentionLayer(emb_size * 2)  # 128*2 for bidirectional
        
        self.dense1 = nn.Linear(emb_size * 2, dim_ff, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(dim_ff)
        self.drop1 = nn.Dropout(dropout + 0.2)
        
        self.dense2 = nn.Linear(dim_ff, dim_ff // 2, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(dim_ff // 2)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x, pad_mask=None):
        # input is (bs, 1, T, C)
        x = x.permute(0, 1, 3, 2) # (bs, 1, C, T)
        imu = x.squeeze(1) # (bs, C, T)

        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        
        merged = x1.transpose(1, 2) # (batch, seq_len, 128)
        
        gru_out, _ = self.bigru(merged)
        gru_out = self.gru_dropout(gru_out)
        
        attended = self.attention(gru_out)
        
        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        return x # [batch, dim_ff // 2] = [batch, 128]

class FilterNetFeatureExtractor(FilterNet_SingleSensor_v1):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.end_stacks = nn.ModuleList()
    
    def _forward(self, X, pad_mask=None):
        X = X[:, 0, :, :].transpose(1, 2)          # [B, C, T]
        Xs = [X]
        Xs.append(self.down_stack_1(Xs[-1]))

        to_merge = [Xs[-1]]
        for module in self.down_stack_2:
            out = module(Xs[-1])
            Xs.append(out)
            to_merge.append(
                F.interpolate(out, size=to_merge[0].shape[-1],
                              mode="linear", align_corners=False)
            )

        merged = torch.cat(to_merge, dim=1)
        Xs.append(merged)
        Xs.append(self.lstm_stack(Xs[-1]))         # [B, hidden_dim, T']

        if self.keep_intermediates:
            self.Xs = Xs

        feats = Xs[-1]                             # [B, hidden_dim, T']
        feats = feats[:, :, -1]                    # [B, hidden_dim]
        return feats

    def forward(self, X, pad_mask=None):
        return self._forward(X, pad_mask)
    
class ConvTran_SingleSensor_NoTranLol_Extractor(nn.Module):
    def __init__(self, 
                 channel_size=cfg.imu_vars, 
                 seq_len=cfg.seq_len, 
                 emb_size=cfg.convtran_emb_size, 
                 num_heads=cfg.convtran_num_heads, 
                 dim_ff=cfg.convtran_dim_ff, 
                 dropout=cfg.convtran_dropout):
        super().__init__()

        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, emb_size * 4, kernel_size=[1, 15], padding='same'),
            nn.BatchNorm2d(emb_size * 4),
            nn.GELU()
        )
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=[channel_size, 1],
                      padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU()
        )

        self.conv_extra = nn.Sequential(
            nn.Conv1d(emb_size, emb_size, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(emb_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.se = SEBlock228(emb_size, r=16)

        self.ffn = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(emb_size, eps=1e-5)

        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, pad_mask=None):
        # x: (bs, 1, T, C) → (bs, 1, C, T)
        x = x.permute(0, 1, 3, 2)

        x = self.embed_layer(x)
        x = self.embed_layer2(x).squeeze(2)         # (bs, emb, T)

        x = self.conv_extra(x)
        x = self.se(x)

        x = x.permute(0, 2, 1)                      # (bs, T, emb)
        x = self.norm(x + self.ffn(x))              # residual + FFN
        x = x.permute(0, 2, 1)                      # (bs, emb, T)

        x = self.gap(x).squeeze(-1)                 # (bs, emb) = (bs, 64)

        return x
    
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

class Public2_SingleSensor_Extractor(nn.Module):
    def __init__(
        self,
        channel_size=cfg.imu_vars, 
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

        fused = torch.cat([pooled_output, meta_proj], dim=1) # 256 + 32 = 288

        return fused
    
class MultiResidualBiGRU_SingleSensor_Extractor(nn.Module):
    def __init__(self, 
                 seq_len=cfg.seq_len,
                 n_imu_vars=cfg.imu_vars,
                 hidden_size=128, 
                 n_layers=3, 
                 bidir=True,
                 dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.n_imu_vars = n_imu_vars
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.imu_input_projection = nn.Linear(n_imu_vars, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        self.res_bigrus = nn.ModuleList([
            ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir)
            for _ in range(n_layers)
        ])
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        pooled_size = hidden_size * 2
        self.feature_fusion = nn.Sequential(
            nn.Linear(pooled_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, imu_data, pad_mask=None, h=None):
        batch_size = imu_data.size(0)
        
        x = imu_data.squeeze(1)
        
        x = self.imu_input_projection(x)  # (B, L, hidden_size)
        x = self.input_ln(x)
        x = nn.functional.relu(x)
        x = self.input_dropout(x)
        
        if h is None:
            h = [None for _ in range(self.n_layers)]
        
        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)
        
        # x shape: (B, L, hidden_size) -> (B, hidden_size, L) for pooling
        x_transposed = x.transpose(1, 2)
        
        avg_pooled = self.global_avg_pool(x_transposed).squeeze(-1)  # (B, hidden_size)
        max_pooled = self.global_max_pool(x_transposed).squeeze(-1)  # (B, hidden_size)
        
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)  # (B, hidden_size*2)
        features = self.feature_fusion(pooled_features)  # (B, hidden_size)
        
        return features
    
# class HybridModel_SingleSensor_v1(nn.Module):
#     def __init__(self, 
#                  channel_size=cfg.imu_vars, 
#                  emb_size_public=128,
#                  dim_ff_public=256,
#                  dropout_public=0.3,
#                  convtran_emb_size=cfg.convtran_emb_size,
#                  convtran_num_heads=cfg.convtran_num_heads, 
#                  convtran_dim_ff=cfg.convtran_dim_ff, 
#                  convtran_dropout=cfg.convtran_dropout,
#                  out_size_public2=256+32,
#                  cnn1d_out_channels=32,
#                  multibigru_dim=128,
#                  multibigru_layers=3,
#                  multibigru_dropout=0.1,
#                  seq_len=cfg.seq_len,
#                  head_droupout=0.2,
#                  num_classes=cfg.main_num_classes):
#         super().__init__()

#         self.extractor1 = Public_SingleSensor_Extractor(
#             channel_size=channel_size, 
#             emb_size=emb_size_public, 
#             dim_ff=dim_ff_public, 
#             dropout=dropout_public
#         ) # output dim = dim_ff_public // 2 = 64

#         self.extractor2 = FilterNetFeatureExtractor() # output dim = DEFAULT_WIDTH = 100

#         self.extractor3 = ConvTran_SingleSensor_NoTranLol_Extractor(
#             channel_size=channel_size, 
#             seq_len=seq_len, 
#             emb_size=convtran_emb_size, 
#             num_heads=convtran_num_heads, 
#             dim_ff=convtran_dim_ff, 
#             dropout=convtran_dropout
#         ) # output dim = convtran_emb_size = 64

#         self.extractor4 = Resnet1DFeatureExtractor(
#             n_in_channels=channel_size, out_channels=cnn1d_out_channels
#         ) # output dim = cnn1d_out_channels * 4 = 32 * 4 = 128
#         self.pool4 = SEPlusMean(cnn1d_out_channels * 4)
#         self.neck4 = MLPNeck(cnn1d_out_channels * 4)

#         self.extractor5 = Public2_SingleSensor_Extractor(channel_size=channel_size)

#         self.extractor6 = MultiResidualBiGRU_SingleSensor_Extractor(
#             seq_len=seq_len,
#             n_imu_vars=channel_size,
#             hidden_size=multibigru_dim, 
#             n_layers=multibigru_layers, 
#             bidir=True,
#             dropout=multibigru_dropout
#         )

#         # 64 + 100 + 64 + 128 + 288 + 128 = 772
#         general_hdim = (dim_ff_public // 2) + (DEFAULT_WIDTH) + (convtran_emb_size) + (cnn1d_out_channels * 4) + (out_size_public2) + (multibigru_dim)

#         self.head1 = nn.Sequential(
#             nn.Linear(general_hdim, general_hdim),
#             nn.ReLU(),
#             nn.Dropout(head_droupout),
#             nn.Linear(general_hdim, num_classes)
#         ) 

#         self.head2 = nn.Sequential(
#             nn.Linear(general_hdim, general_hdim),
#             nn.ReLU(),
#             nn.Dropout(head_droupout),
#             nn.Linear(general_hdim, 2)
#         ) 

#         self.head3 = nn.Sequential(
#             nn.Linear(general_hdim, general_hdim),
#             nn.ReLU(),
#             nn.Dropout(head_droupout),
#             nn.Linear(general_hdim, 4)
#         ) 

#     def forward(self, x, pad_mask=None):
#         # input is (bs, 1, T, C)

#         x1 = self.extractor1(x) # (bs, dim_ff_public // 2)
#         x2 = self.extractor2(x) # (bs, DEFAULT_WIDTH)
#         x3 = self.extractor3(x) # (bs, convtran_emb_size)

#         x_ = x.permute(0, 1, 3, 2) # (bs, 1, C, T)
#         x_ = x_.squeeze(1) # (bs, C, T)

#         x4 = self.extractor4(x_) # (bs, cnn1d_out_channels * 4, T)
#         x4 = self.pool4(x4) # (bs, cnn1d_out_channels * 4)
#         x4 = x4 + self.neck4(x4) # (bs, cnn1d_out_channels * 4)

#         x5 = self.extractor5(x) # (bs, 256 + 32)

#         x6 = self.extractor6(x) # (bs, multibigru_dim)

#         x_cat = torch.cat([x1, x2, x3, x4, x5, x6], dim=1) # (bs, general_hdim)

#         out1 = self.head1(x_cat) # (bs, num_classes)
#         out2 = self.head2(x_cat) # (bs, 2)
#         out3 = self.head3(x_cat) # (bs, 4)

#         return out1, out2, out3

# class HybridModel_SingleSensor_v1(nn.Module):
#     def __init__(self, 
#                  final_hidden_dim=256,
#                  emb_size_public=128,
#                  dim_ff_public=256,
#                  dropout_public=0.3,
#                  convtran_emb_size=cfg.convtran_emb_size,
#                  convtran_num_heads=cfg.convtran_num_heads, 
#                  convtran_dim_ff=cfg.convtran_dim_ff, 
#                  convtran_dropout=cfg.convtran_dropout,
#                  out_size_public2=256+32,
#                  cnn1d_out_channels=32,
#                  multibigru_dim=128,
#                  multibigru_layers=3,
#                  multibigru_dropout=0.1,
#                  seq_len=cfg.seq_len,
#                  head_droupout=0.2,
#                  num_classes=cfg.main_num_classes):
#         super().__init__()
        
#         self.channel_sizes = {
#             'imu': 3,      # x_imu: 0-2
#             'rot': 4,      # x_rot: 3-6  
#             'fe1': 13,     # x_fe1: 7-19
#             'fe2': 9,      # x_fe2: 20-28
#             'full': 29     # x_full: 0-28
#         }
        
#         self.branch_extractors = nn.ModuleDict()
        
#         for branch_name, channel_size in self.channel_sizes.items():
#             self.branch_extractors[f'{branch_name}_extractor1'] = Public_SingleSensor_Extractor(
#                 channel_size=channel_size, 
#                 emb_size=emb_size_public, 
#                 dim_ff=dim_ff_public, 
#                 dropout=dropout_public
#             )
            
#             self.branch_extractors[f'{branch_name}_extractor2'] = FilterNetFeatureExtractor(input_channels=channel_size)
            
#             self.branch_extractors[f'{branch_name}_extractor3'] = ConvTran_SingleSensor_NoTranLol_Extractor(
#                 channel_size=channel_size, 
#                 seq_len=seq_len, 
#                 emb_size=convtran_emb_size, 
#                 num_heads=convtran_num_heads, 
#                 dim_ff=convtran_dim_ff, 
#                 dropout=convtran_dropout
#             )
            
#             self.branch_extractors[f'{branch_name}_extractor4'] = Resnet1DFeatureExtractor(
#                 n_in_channels=channel_size, out_channels=cnn1d_out_channels
#             )
#             self.branch_extractors[f'{branch_name}_pool4'] = SEPlusMean(cnn1d_out_channels * 4)
#             self.branch_extractors[f'{branch_name}_neck4'] = MLPNeck(cnn1d_out_channels * 4)
            
#             self.branch_extractors[f'{branch_name}_extractor5'] = Public2_SingleSensor_Extractor(
#                 channel_size=channel_size
#             )
            
#             self.branch_extractors[f'{branch_name}_extractor6'] = MultiResidualBiGRU_SingleSensor_Extractor(
#                 seq_len=seq_len,
#                 n_imu_vars=channel_size,
#                 hidden_size=multibigru_dim, 
#                 n_layers=multibigru_layers, 
#                 bidir=True,
#                 dropout=multibigru_dropout
#             )
        
#         # 64 + 100 + 64 + 128 + 288 + 128 = 772
#         branch_feature_dim = (dim_ff_public // 2) + (DEFAULT_WIDTH) + (convtran_emb_size) + \
#                            (cnn1d_out_channels * 4) + (out_size_public2) + (multibigru_dim)
        
#         self.branch_projections = nn.ModuleDict()
#         for branch_name in self.channel_sizes.keys():
#             self.branch_projections[f'{branch_name}_projection'] = nn.Sequential(
#                 nn.Linear(branch_feature_dim, final_hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(head_droupout),
#                 nn.BatchNorm1d(final_hidden_dim)
#             )
        
#         final_feature_dim = final_hidden_dim * 5
        
#         self.head1 = nn.Sequential(
#             nn.Linear(final_feature_dim, final_feature_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(head_droupout),
#             nn.Linear(final_feature_dim // 2, num_classes)
#         )

#         self.head2 = nn.Sequential(
#             nn.Linear(final_feature_dim, final_feature_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(head_droupout),
#             nn.Linear(final_feature_dim // 2, 2)
#         )

#         self.head3 = nn.Sequential(
#             nn.Linear(final_feature_dim, final_feature_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(head_droupout),
#             nn.Linear(final_feature_dim // 2, 4)
#         )
            
#     def process_branch(self, x, branch_name):
#         x1 = self.branch_extractors[f'{branch_name}_extractor1'](x)
#         x2 = self.branch_extractors[f'{branch_name}_extractor2'](x)
#         x3 = self.branch_extractors[f'{branch_name}_extractor3'](x)
        
#         x_ = x.permute(0, 1, 3, 2).squeeze(1)  # (bs, C, T)
#         x4 = self.branch_extractors[f'{branch_name}_extractor4'](x_)
#         x4 = self.branch_extractors[f'{branch_name}_pool4'](x4)
#         x4 = x4 + self.branch_extractors[f'{branch_name}_neck4'](x4)
        
#         x5 = self.branch_extractors[f'{branch_name}_extractor5'](x)
#         x6 = self.branch_extractors[f'{branch_name}_extractor6'](x)
        
#         x_cat = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        
#         projected = self.branch_projections[f'{branch_name}_projection'](x_cat)
        
#         return projected
    
#     def forward(self, _x, pad_mask=None):
#         # input is (bs, 1, T, C)
        
#         x_imu = _x[:, :, :, :3]
#         x_rot = _x[:, :, :, 3:7] 
#         x_fe1 = _x[:, :, :, 7:20]
#         x_fe2 = _x[:, :, :, 20:29]
#         x_full = _x
        
#         imu_features = self.process_branch(x_imu, 'imu')
#         rot_features = self.process_branch(x_rot, 'rot')
#         fe1_features = self.process_branch(x_fe1, 'fe1')
#         fe2_features = self.process_branch(x_fe2, 'fe2')
#         full_features = self.process_branch(x_full, 'full')
        
#         final_features = torch.cat([
#             imu_features, rot_features, fe1_features, fe2_features, full_features
#         ], dim=1)
        
#         out1 = self.head1(final_features)
#         out2 = self.head2(final_features)
#         out3 = self.head3(final_features)

#         return out1, out2, out3
    
class HybridModel_SingleSensor_v1(nn.Module):
    def __init__(self, 
                 final_hidden_dim=256,
                 emb_size_public=128,
                 dim_ff_public=256,
                 dropout_public=0.3,
                 convtran_emb_size=cfg.convtran_emb_size,
                 convtran_num_heads=cfg.convtran_num_heads, 
                 convtran_dim_ff=cfg.convtran_dim_ff, 
                 convtran_dropout=cfg.convtran_dropout,
                 out_size_public2=256+32,
                 cnn1d_out_channels=32,
                 multibigru_dim=128,
                 multibigru_layers=3,
                 multibigru_dropout=0.1,
                 seq_len=cfg.seq_len,
                 head_droupout=0.2,
                 num_classes=cfg.main_num_classes):
        super().__init__()
        
        self.channel_sizes = {
            'imu': 3,      # x_imu: 0-2
            'rot': 4,      # x_rot: 3-6  
            'fe1': 13+3+3,     # x_fe1: 7-19+3+3
            'fe2': 9,      # x_fe2: 20+3+3-28+3+3
            'full': 29+3+3     # x_full: 0-28+3+3
        }
        
        self.branch_extractors = nn.ModuleDict()
        
        for branch_name, channel_size in self.channel_sizes.items():
            self.branch_extractors[f'{branch_name}_extractor1'] = Public_SingleSensor_Extractor(
                channel_size=channel_size, 
                emb_size=emb_size_public, 
                dim_ff=dim_ff_public, 
                dropout=dropout_public
            )
            
            self.branch_extractors[f'{branch_name}_extractor2'] = FilterNetFeatureExtractor(input_channels=channel_size)
            
            self.branch_extractors[f'{branch_name}_extractor3'] = ConvTran_SingleSensor_NoTranLol_Extractor(
                channel_size=channel_size, 
                seq_len=seq_len, 
                emb_size=convtran_emb_size, 
                num_heads=convtran_num_heads, 
                dim_ff=convtran_dim_ff, 
                dropout=convtran_dropout
            )
            
            self.branch_extractors[f'{branch_name}_extractor4'] = Resnet1DFeatureExtractor(
                n_in_channels=channel_size, out_channels=cnn1d_out_channels
            )
            self.branch_extractors[f'{branch_name}_pool4'] = SEPlusMean(cnn1d_out_channels * 4)
            self.branch_extractors[f'{branch_name}_neck4'] = MLPNeck(cnn1d_out_channels * 4)
            
            self.branch_extractors[f'{branch_name}_extractor5'] = Public2_SingleSensor_Extractor(
                channel_size=channel_size
            )
            
            self.branch_extractors[f'{branch_name}_extractor6'] = MultiResidualBiGRU_SingleSensor_Extractor(
                seq_len=seq_len,
                n_imu_vars=channel_size,
                hidden_size=multibigru_dim, 
                n_layers=multibigru_layers, 
                bidir=True,
                dropout=multibigru_dropout
            )
        
        extractor_feature_dims = {
            'extractor1': (dim_ff_public // 2) * 5,
            'extractor2': (DEFAULT_WIDTH) * 5,
            'extractor3': (convtran_emb_size) * 5,
            'extractor4': (cnn1d_out_channels * 4) * 5,
            'extractor5': (out_size_public2) * 5,
            'extractor6': (multibigru_dim) * 5
        }
        
        self.extractor_projections = nn.ModuleDict()
        for extractor_name, feature_dim in extractor_feature_dims.items():
            self.extractor_projections[f'{extractor_name}_projection'] = nn.Sequential(
                nn.Linear(feature_dim, final_hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_droupout),
                nn.BatchNorm1d(final_hidden_dim)
            )
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=final_hidden_dim,
            num_heads=8,
            dropout=head_droupout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(final_hidden_dim)

        final_feature_dim = final_hidden_dim * 6

        self.head1 = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_feature_dim // 2, num_classes)
        )

        self.head2 = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_feature_dim // 2, 2)
        )

        self.head3 = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_feature_dim // 2, 4)
        )

        self.ext1_head1 = nn.Sequential(
            nn.Linear(final_hidden_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_hidden_dim, num_classes)
        ) 
        self.ext2_head1 = nn.Sequential(
            nn.Linear(final_hidden_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_hidden_dim, num_classes)
        ) 
        self.ext3_head1 = nn.Sequential(
            nn.Linear(final_hidden_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_hidden_dim, num_classes)
        ) 
        self.ext4_head1 = nn.Sequential(
            nn.Linear(final_hidden_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_hidden_dim, num_classes)
        ) 
        self.ext5_head1 = nn.Sequential(
            nn.Linear(final_hidden_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_hidden_dim, num_classes)
        )
        self.ext6_head1 = nn.Sequential(
            nn.Linear(final_hidden_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(final_hidden_dim, num_classes)
        )
            
    def process_extractor(self, x_dict, extractor_num):
        extractor_name = f'extractor{extractor_num}'
        branch_features = []
        
        for branch_name in self.channel_sizes.keys():
            x = x_dict[branch_name]
            
            if extractor_num == 1:
                feature = self.branch_extractors[f'{branch_name}_extractor1'](x)
            elif extractor_num == 2:
                feature = self.branch_extractors[f'{branch_name}_extractor2'](x)
            elif extractor_num == 3:
                feature = self.branch_extractors[f'{branch_name}_extractor3'](x)
            elif extractor_num == 4:
                x_ = x.permute(0, 1, 3, 2).squeeze(1)  # (bs, C, T)
                feature = self.branch_extractors[f'{branch_name}_extractor4'](x_)
                feature = self.branch_extractors[f'{branch_name}_pool4'](feature)
                feature = feature + self.branch_extractors[f'{branch_name}_neck4'](feature)
            elif extractor_num == 5:
                feature = self.branch_extractors[f'{branch_name}_extractor5'](x)
            elif extractor_num == 6:
                feature = self.branch_extractors[f'{branch_name}_extractor6'](x)
            
            branch_features.append(feature)
        
        x_cat = torch.cat(branch_features, dim=1)
        
        projected = self.extractor_projections[f'{extractor_name}_projection'](x_cat)
        
        return projected
    
    def forward(self, _x, pad_mask=None):
        # input is (bs, 1, T, C)
        
        x_dict = {
            'imu': _x[:, :, :, :3],
            'rot': _x[:, :, :, 3:7],
            'fe1': _x[:, :, :, 7:20+3+3],
            'fe2': _x[:, :, :, 20+3+3:29+3+3],
            'full': _x
        }
        
        extractor_features = []
        for extractor_num in range(1, 7):
            feature = self.process_extractor(x_dict, extractor_num)
            extractor_features.append(feature)
        
        stacked_features = torch.stack(extractor_features, dim=1)  # (bs, 6, final_hidden_dim)
        
        attended_features, _ = self.self_attention(
            stacked_features, 
            stacked_features, 
            stacked_features
        )  # (bs, 6, final_hidden_dim)
        
        attended_features = self.attention_norm(attended_features + stacked_features)
        
        final_features = attended_features.view(attended_features.size(0), -1)  # (bs, final_hidden_dim * 6)
        
        # final_features = torch.cat(extractor_features, dim=1)
        
        out1 = self.head1(final_features)
        out2 = self.head2(final_features)
        out3 = self.head3(final_features)

        ext1_out1 = self.ext1_head1(extractor_features[0])
        ext2_out1 = self.ext2_head1(extractor_features[1])
        ext3_out1 = self.ext3_head1(extractor_features[2])
        ext4_out1 = self.ext4_head1(extractor_features[3])
        ext5_out1 = self.ext5_head1(extractor_features[4])
        ext6_out1 = self.ext6_head1(extractor_features[5])

        return out1, out2, out3, ext1_out1, ext2_out1, ext3_out1, ext4_out1, ext5_out1, ext6_out1