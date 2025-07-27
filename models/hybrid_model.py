import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct, idct

from modules.inceptiontime_replacers import Resnet1DFeatureExtractor
from models.convtran import SEBlock228
from models.filternet import FilterNet_SingleSensor_Test, DEFAULT_WIDTH
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

class FilterNetFeatureExtractor(FilterNet_SingleSensor_Test):
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
    def forward(self, x, pad_mask=None):
        # x shape: (B, L, C)
        
        if pad_mask is None:
            mean = torch.mean(x, dim=1)
            std = torch.std(x, dim=1)
            max_val, _ = torch.max(x, dim=1)
            min_val, _ = torch.min(x, dim=1)
            
            seq_len = x.size(1)
            if seq_len > 1:
                slope = (x[:, -1, :] - x[:, 0, :]) / (seq_len - 1)
            else:
                slope = torch.zeros_like(x[:, 0, :])
        else:
            mask = pad_mask.unsqueeze(-1)  # (B, L, 1)
            
            masked_x = x * mask
            valid_count = mask.sum(dim=1).clamp_min(1)  # (B, 1)
            mean = masked_x.sum(dim=1) / valid_count
            
            diff_squared = ((x - mean.unsqueeze(1)) ** 2) * mask
            variance = diff_squared.sum(dim=1) / valid_count.clamp_min(2)
            std = torch.sqrt(variance.clamp_min(1e-8))
            
            masked_x_max = x.clone()
            masked_x_max[mask.squeeze(-1) == 0] = float('-inf')
            max_val, _ = torch.max(masked_x_max, dim=1)
            
            masked_x_min = x.clone()
            masked_x_min[mask.squeeze(-1) == 0] = float('inf')
            min_val, _ = torch.min(masked_x_min, dim=1)
            
            valid_indices = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
            valid_indices = valid_indices * pad_mask
            
            first_valid_idx = torch.argmax(pad_mask.float(), dim=1)  # (B,)
            
            last_valid_idx = x.size(1) - 1 - torch.argmax(pad_mask.flip(1).float(), dim=1)  # (B,)
            
            batch_indices = torch.arange(x.size(0), device=x.device)
            first_vals = x[batch_indices, first_valid_idx]  # (B, C)
            last_vals = x[batch_indices, last_valid_idx]    # (B, C)
            
            seq_len_valid = (last_valid_idx - first_valid_idx).float().unsqueeze(-1).clamp_min(1)  # (B, 1)
            slope = (last_vals - first_vals) / seq_len_valid
            
            single_valid = (pad_mask.sum(dim=1) == 1).unsqueeze(-1)  # (B, 1)
            slope = slope * (~single_valid).float()
        
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

        meta = self.meta_extractor(x, pad_mask=pad_mask)
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

class DCTMaskBlock(nn.Module):
    def __init__(self, seq_len, n_channels, init_kernels=None):
        super().__init__()
        self.seq_len = seq_len
        self.conv = nn.Conv1d(1, n_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.conv.weight, 1.0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        bs, C, T = x.shape
        x_dct = dct(x, norm='ortho')
        ones = torch.ones(bs, 1, T, device=x.device)
        w = self.act(self.conv(ones))
        x_dct = x_dct * w
        x_idct = idct(x_dct, norm='ortho') # / (T * 2.0)
        return x_idct.permute(0, 2, 1)
    
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
                 attention_n_heads=8,
                 attention_dropout=0.2,
                 use_dct=True,
                 num_classes=cfg.main_num_classes):
        super().__init__()
        
        self.use_dct = use_dct
        if self.use_dct:
            self.dct_bp = DCTMaskBlock(seq_len=seq_len, n_channels=32)
        
        self.channel_sizes = {
            'imu': 3,      # x_imu: 0-2
            'rot': 4,      # x_rot: 3-6  
            'fe1': 13+3,     # x_fe1: 7-19+3
            'fe2': 9,      # x_fe2: 20+3-28+3
            'full': 29+3     # x_full: 0-28+3
        }
        
        self.branch_extractors = nn.ModuleDict()

        for branch_name, channel_size in self.channel_sizes.items():
            self.branch_extractors[f'{branch_name}_extractor1'] = Public_SingleSensor_Extractor(
                channel_size=channel_size, 
                emb_size=emb_size_public, 
                dim_ff=dim_ff_public, 
                dropout=dropout_public
            )
            
            self.branch_extractors[f'{branch_name}_extractor2'] = FilterNetFeatureExtractor(
                input_channels=channel_size,
                do_multi=False
            )
            
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
            num_heads=attention_n_heads,
            dropout=attention_dropout,
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
            
    def process_extractor(self, x_dict, extractor_num, pad_mask=None):
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
                feature = self.branch_extractors[f'{branch_name}_extractor5'](x, pad_mask=pad_mask)
            elif extractor_num == 6:
                feature = self.branch_extractors[f'{branch_name}_extractor6'](x)
            
            branch_features.append(feature)
        
        x_cat = torch.cat(branch_features, dim=1)
        
        projected = self.extractor_projections[f'{extractor_name}_projection'](x_cat)
        
        return projected
    
    def forward(self, _x, pad_mask=None):
        # input is (bs, 1, T, C)

        if self.use_dct:
            x_wave = _x.squeeze(1).permute(0, 2, 1)
            x_wave = self.dct_bp(x_wave)
            _x = x_wave.unsqueeze(1)

        x_dict = {
            'imu': _x[:, :, :, :3],
            'rot': _x[:, :, :, 3:7],
            'fe1': _x[:, :, :, 7:20+3],
            'fe2': _x[:, :, :, 20+3:29+3],
            'full': _x
        }
        
        extractor_features = []
        for extractor_num in range(1, 7):
            feature = self.process_extractor(x_dict, extractor_num, pad_mask=pad_mask)
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
    
class MultiSensor_HybridModel_v1(nn.Module):
    def __init__(self, 
                 final_hidden_dim=256,
                 emb_size_public=128,
                 dim_ff_public=256,
                 dropout_public=0.3,
                 cnn1d_out_channels=32,
                 multibigru_dim=128,
                 multibigru_layers=3,
                 multibigru_dropout=0.1,
                 seq_len=cfg.seq_len,
                 head_droupout=0.2,
                 attention_n_heads=8,
                 attention_dropout=0.2,
                 num_classes=cfg.main_num_classes):
        super().__init__()
        
        self.channel_sizes = {
            'imu': 3,      # x_imu: 0-2
            'rot': 4,      # x_rot: 3-6  
            'fe1': 13+3,     # x_fe1: 7-19+3
            'fe2': 9,      # x_fe2: 20+3-28+3
            'full': 29+3,     # x_full: 0-28+3
            'thm': 5,
            'tof1': 64,
            'tof2': 64,
            'tof3': 64,
            'tof4': 64,
            'tof5': 64,
        }
        
        self.branch_extractors = nn.ModuleDict()
        
        for branch_name, channel_size in self.channel_sizes.items():
            self.branch_extractors[f'{branch_name}_extractor1'] = Public_SingleSensor_Extractor(
                channel_size=channel_size, 
                emb_size=emb_size_public, 
                dim_ff=dim_ff_public, 
                dropout=dropout_public
            )
            
            self.branch_extractors[f'{branch_name}_extractor2'] = FilterNetFeatureExtractor(
                input_channels=channel_size,
                do_multi=False
            )
            
            self.branch_extractors[f'{branch_name}_extractor3'] = Resnet1DFeatureExtractor(
                n_in_channels=channel_size, out_channels=cnn1d_out_channels
            )
            self.branch_extractors[f'{branch_name}_pool3'] = SEPlusMean(cnn1d_out_channels * 4)
            self.branch_extractors[f'{branch_name}_neck3'] = MLPNeck(cnn1d_out_channels * 4)
            
            self.branch_extractors[f'{branch_name}_extractor4'] = MultiResidualBiGRU_SingleSensor_Extractor(
                seq_len=seq_len,
                n_imu_vars=channel_size,
                hidden_size=multibigru_dim, 
                n_layers=multibigru_layers, 
                bidir=True,
                dropout=multibigru_dropout
            )
        
        extractor_feature_dims = {
            'extractor1': (dim_ff_public // 2) * 11,
            'extractor2': (DEFAULT_WIDTH) * 11,
            'extractor3': (cnn1d_out_channels * 4) * 11,
            'extractor4': (multibigru_dim) * 11
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
            num_heads=attention_n_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(final_hidden_dim)

        final_feature_dim = final_hidden_dim * 4

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
            
    def process_extractor(self, x_dict, extractor_num, pad_mask=None):
        extractor_name = f'extractor{extractor_num}'
        branch_features = []
        
        for branch_name in self.channel_sizes.keys():
            x = x_dict[branch_name]
            
            if extractor_num == 1:
                feature = self.branch_extractors[f'{branch_name}_extractor1'](x)
            elif extractor_num == 2:
                feature = self.branch_extractors[f'{branch_name}_extractor2'](x)
            elif extractor_num == 3:
                x_ = x.permute(0, 1, 3, 2).squeeze(1)  # (bs, C, T)
                feature = self.branch_extractors[f'{branch_name}_extractor3'](x_)
                feature = self.branch_extractors[f'{branch_name}_pool3'](feature)
                feature = feature + self.branch_extractors[f'{branch_name}_neck3'](feature)
            elif extractor_num == 4:
                feature = self.branch_extractors[f'{branch_name}_extractor4'](x)
            
            branch_features.append(feature)
        
        x_cat = torch.cat(branch_features, dim=1)
        
        projected = self.extractor_projections[f'{extractor_name}_projection'](x_cat)
        
        return projected
    
    def forward(self, _x, thm, tof, pad_mask=None):
        # input is (bs, 1, T, C)
        
        x_dict = {
            'imu': _x[:, :, :, :3],
            'rot': _x[:, :, :, 3:7],
            'fe1': _x[:, :, :, 7:20+3],
            'fe2': _x[:, :, :, 20+3:29+3],
            'full': _x,
            'thm': thm.permute(0, 3, 2, 1),
            'tof1': tof[:, 0:1, :, :],
            'tof2': tof[:, 1:2, :, :],
            'tof3': tof[:, 2:3, :, :],
            'tof4': tof[:, 3:4, :, :],
            'tof5': tof[:, 4:5, :, :],
        }
        
        extractor_features = []
        for extractor_num in range(1, 5):
            feature = self.process_extractor(x_dict, extractor_num, pad_mask=pad_mask)
            extractor_features.append(feature)

        stacked_features = torch.stack(extractor_features, dim=1)  # (bs, 4, final_hidden_dim)
        
        attended_features, _ = self.self_attention(
            stacked_features, 
            stacked_features, 
            stacked_features
        )  # (bs, 4, final_hidden_dim)
        
        attended_features = self.attention_norm(attended_features + stacked_features)
        
        final_features = attended_features.view(attended_features.size(0), -1)  # (bs, final_hidden_dim * 4)
        
        out1 = self.head1(final_features)
        out2 = self.head2(final_features)
        out3 = self.head3(final_features)

        ext1_out1 = self.ext1_head1(extractor_features[0])
        ext2_out1 = self.ext2_head1(extractor_features[1])
        ext3_out1 = self.ext3_head1(extractor_features[2])
        ext4_out1 = self.ext4_head1(extractor_features[3])

        return out1, out2, out3, ext1_out1, ext2_out1, ext3_out1, ext4_out1