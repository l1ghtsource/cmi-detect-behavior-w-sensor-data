import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.inceptiontime import InceptionTimeFeatureExtractor
from modules.inceptiontime_replacers import (
    Resnet1DFeatureExtractor, 
    EfficientNet1DFeatureExtractor, 
    DenseNet1DFeatureExtractor
)
from models.convtran import SEBlock228
from models.filternet import FilterNet_SingleSensor_v1, DEFAULT_WIDTH
from models.basic_cnn1ds import SEPlusMean, MLPNeck
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
    
class HybridModel_SingleSensor_v1(nn.Module):
    def __init__(self, 
                 channel_size=cfg.imu_vars, 
                 emb_size_public=128,
                 dim_ff_public=256,
                 dropout_public=0.3,
                 convtran_emb_size=cfg.convtran_emb_size,
                 convtran_num_heads=cfg.convtran_num_heads, 
                 convtran_dim_ff=cfg.convtran_dim_ff, 
                 convtran_dropout=cfg.convtran_dropout,
                 cnn1d_out_channels=32,
                 seq_len=cfg.seq_len,
                 head_droupout=0.2,
                 num_classes=cfg.main_num_classes):
        super().__init__()

        self.extractor1 = Public_SingleSensor_Extractor(
            channel_size=channel_size, 
            emb_size=emb_size_public, 
            dim_ff=dim_ff_public, 
            dropout=dropout_public
        ) # output dim = dim_ff_public // 2 = 64

        self.extractor2 = FilterNetFeatureExtractor() # output dim = DEFAULT_WIDTH = 100

        self.extractor3 = ConvTran_SingleSensor_NoTranLol_Extractor(
            channel_size=channel_size, 
            seq_len=seq_len, 
            emb_size=convtran_emb_size, 
            num_heads=convtran_num_heads, 
            dim_ff=convtran_dim_ff, 
            dropout=convtran_dropout
        ) # output dim = convtran_emb_size = 64

        self.extractor4 = Resnet1DFeatureExtractor(
            n_in_channels=channel_size, out_channels=cnn1d_out_channels
        ) # output dim = cnn1d_out_channels * 4 = 32 * 4 = 128

        self.pool = SEPlusMean(cnn1d_out_channels * 4)
        self.neck = MLPNeck(cnn1d_out_channels * 4)

        general_hdim = (dim_ff_public // 2) + (DEFAULT_WIDTH) + (convtran_emb_size) + (cnn1d_out_channels * 4) # 64 + 100 + 64 + 128 = 356

        self.head1 = nn.Sequential(
            nn.Linear(general_hdim, general_hdim),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(general_hdim, num_classes)
        ) 

        self.head2 = nn.Sequential(
            nn.Linear(general_hdim, general_hdim),
            nn.ReLU(),
            nn.Dropout(head_droupout),
            nn.Linear(general_hdim, 2)
        ) 

    def forward(self, x, pad_mask=None):
        # input is (bs, 1, T, C)

        x1 = self.extractor1(x) # (bs, dim_ff_public // 2)
        x2 = self.extractor2(x) # (bs, DEFAULT_WIDTH)
        x3 = self.extractor3(x) # (bs, convtran_emb_size)

        x_ = x.permute(0, 1, 3, 2) # (bs, 1, C, T)
        x_ = x_.squeeze(1) # (bs, C, T)

        x4 = self.extractor4(x_) # (bs, cnn1d_out_channels * 4, T)
        x4 = self.pool(x4) # (bs, cnn1d_out_channels * 4)
        x4 = x4 + self.neck(x4) # (bs, cnn1d_out_channels * 4)

        x_cat = torch.cat([x1, x2, x3, x4], dim=1) # (bs, general_hdim)

        out1 = self.head1(x_cat) # (bs, num_classes)
        out2 = self.head2(x_cat) # (bs, 2)

        return out1, out2