import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange

from modules.inceptiontime import InceptionTimeFeatureExtractor
from configs.config import cfg

# good in hybrid, solo ok

class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)

class Attention_Rel_Scl(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.num_heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        # distance_pd = pd.DataFrame(relative_bias[0,0,:,:].cpu().detach().numpy())
        # distance_pd.to_csv('scalar_position_distance.csv')

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out
    
class ResidualEmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)
    
class SE_Block(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBlock228(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)          # squeeze
        self.fc = nn.Sequential(                     # excitation
            nn.Conv1d(channels, channels // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // r, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):                           # x: (bs, C, T)
        w = self.pool(x)                            # (bs, C, 1)
        w = self.fc(w)                              # (bs, C, 1)
        return x * w                                # channel-wise re-weighting

class ConvTran_SingleSensor_v1(nn.Module):
    def __init__(self, 
                 channel_size=cfg.imu_vars, 
                 seq_len=cfg.seq_len, 
                 emb_size=cfg.convtran_emb_size, 
                 num_heads=cfg.convtran_num_heads, 
                 dim_ff=cfg.convtran_dim_ff, 
                 dropout=cfg.convtran_dropout, 
                 num_classes=cfg.main_num_classes):
        super().__init__()

        if seq_len > 128:
            m = 2 ** int(math.log2(seq_len // 128))
        else:
            m = 1

        seq_len = seq_len // m

        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 15], padding='same'),
                                         nn.BatchNorm2d(emb_size*4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        self.maxpool = nn.MaxPool2d(kernel_size=(1, m), stride=(1, m))
        self.Fix_Position = tAPE(emb_size, dropout=dropout, max_len=seq_len)
        self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, dropout)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out1 = nn.Linear(emb_size, num_classes)
        self.out2 = nn.Linear(emb_size, 2)

    def forward(self, x, pad_mask=None):
        # input is (bs, 1, T, C)
        x = x.permute(0, 1, 3, 2) # (bs, 1, C, T)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = self.maxpool(x_src) 
        x_src = x_src.permute(0, 2, 1)
        x_src_pos = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src_pos)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out1 = self.out1(out)
        out2 = self.out2(out)
        return out1, out2
    
class ConvTran_SingleSensor_NoTranLol_v1(nn.Module):
    def __init__(self, 
                 channel_size=cfg.imu_vars, 
                 seq_len=cfg.seq_len, 
                 emb_size=cfg.convtran_emb_size, 
                 num_heads=cfg.convtran_num_heads, 
                 dim_ff=cfg.convtran_dim_ff, 
                 dropout=cfg.convtran_dropout, 
                 num_classes=cfg.main_num_classes):
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
        self.out1 = nn.Linear(emb_size, num_classes)
        self.out2 = nn.Linear(emb_size, 2)

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

        x = self.gap(x).squeeze(-1)                 # (bs, emb)
        cls_logits = self.out1(x)
        reg_logits = self.out2(x)

        return cls_logits, reg_logits

class ConvTran_SingleSensor_SE_v1(nn.Module):
    def __init__(self, 
                 channel_size=cfg.imu_vars, 
                 seq_len=cfg.seq_len, 
                 emb_size=cfg.convtran_emb_size, 
                 num_heads=cfg.convtran_num_heads, 
                 dim_ff=cfg.convtran_dim_ff, 
                 dropout=cfg.convtran_dropout, 
                 num_classes=cfg.main_num_classes):
        super().__init__()

        if seq_len > 128:
            m = 2 ** int(math.log2(seq_len // 128))
        else:
            m = 1

        seq_len = seq_len // m

        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 15], padding='same'),
                                         nn.BatchNorm2d(emb_size*4),
                                         nn.GELU(),
                                         SE_Block(emb_size*4, reduction=8))

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU(),
                                          SE_Block(emb_size, reduction=8))

        self.maxpool = nn.MaxPool2d(kernel_size=(1, m), stride=(1, m))
        self.Fix_Position = tAPE(emb_size, dropout=dropout, max_len=seq_len)
        self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, dropout)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out1 = nn.Linear(emb_size, num_classes)
        self.out2 = nn.Linear(emb_size, 2)

    def forward(self, x, pad_mask=None):
        # input is (bs, 1, T, C)
        x = x.permute(0, 1, 3, 2) # (bs, 1, C, T)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = self.maxpool(x_src) 
        x_src = x_src.permute(0, 2, 1)
        x_src_pos = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src_pos)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out1 = self.out1(out)
        out2 = self.out2(out)
        return out1, out2
    
class ConvTran_SingleSensor_MultiScale_v1(nn.Module):
    def __init__(self, 
                 channel_size=cfg.imu_vars, 
                 seq_len=cfg.seq_len, 
                 emb_size=cfg.convtran_emb_size, 
                 num_heads=cfg.convtran_num_heads, 
                 dim_ff=cfg.convtran_dim_ff, 
                 dropout=cfg.convtran_dropout, 
                 num_classes=cfg.main_num_classes):
        super().__init__()

        if seq_len > 128:
            m = 2 ** int(math.log2(seq_len // 128))
        else:
            m = 1

        seq_len = seq_len // m

        # Multi-scale temporal embedding branches
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, emb_size, kernel_size=[1, k], padding='same'),
                nn.BatchNorm2d(emb_size),
                nn.GELU()
            ) for k in [10, 20, 40]
        ])

        # Fusion of multi-scale features
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(emb_size * 3, emb_size * 4, kernel_size=1),
            nn.BatchNorm2d(emb_size * 4),
            nn.GELU()
        )

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(1, m), stride=(1, m))
        self.Fix_Position = tAPE(emb_size, dropout=dropout, max_len=seq_len)
        self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, dropout)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out1 = nn.Linear(emb_size, num_classes)
        self.out2 = nn.Linear(emb_size, 2)

    def forward(self, x, pad_mask=None):
        # input is (bs, 1, T, C)
        x = x.permute(0, 1, 3, 2)  # (bs, 1, C, T)

        # Apply multi-scale temporal embedding branches
        branch_outputs = [branch(x) for branch in self.temporal_branches]
        x_src = torch.cat(branch_outputs, dim=1)  # concat on channel dim

        x_src = self.temporal_fusion(x_src)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = self.maxpool(x_src)
        x_src = x_src.permute(0, 2, 1)

        x_src_pos = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src_pos)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out1 = self.out1(out)
        out2 = self.out2(out)
        return out1, out2
    
class ConvTran_SingleSensor_Residual_v1(nn.Module):
    def __init__(self, 
                 channel_size=cfg.imu_vars, 
                 seq_len=cfg.seq_len, 
                 emb_size=cfg.convtran_emb_size, 
                 num_heads=cfg.convtran_num_heads, 
                 dim_ff=cfg.convtran_dim_ff, 
                 dropout=cfg.convtran_dropout, 
                 num_classes=cfg.main_num_classes):
        super().__init__()

        if seq_len > 128:
            m = 2 ** int(math.log2(seq_len // 128))
        else:
            m = 1

        seq_len = seq_len // m

        # Residual embedding blocks
        self.embed_blocks = nn.Sequential(
            ResidualEmbeddingBlock(1, emb_size*2, (1, 15)),
            ResidualEmbeddingBlock(emb_size*2, emb_size*4, (1, 9)),
            ResidualEmbeddingBlock(emb_size*4, emb_size*4, (1, 5))
        )

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(1, m), stride=(1, m))
        self.Fix_Position = tAPE(emb_size, dropout=dropout, max_len=seq_len)
        self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, dropout)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out1 = nn.Linear(emb_size, num_classes)
        self.out2 = nn.Linear(emb_size, 2)

    def forward(self, x, pad_mask=None):
        # input is (bs, 1, T, C)
        x = x.permute(0, 1, 3, 2)  # (bs, 1, C, T)

        x_src = self.embed_blocks(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = self.maxpool(x_src)
        x_src = x_src.permute(0, 2, 1)

        x_src_pos = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src_pos)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out1 = self.out1(out)
        out2 = self.out2(out)
        return out1, out2
    
class ConvTran_SingleSensor_Inception_v1(nn.Module):
    def __init__(self, 
                 channel_size=cfg.imu_vars, 
                 seq_len=cfg.seq_len, 
                 emb_size=cfg.convtran_emb_size, 
                 num_heads=cfg.convtran_num_heads, 
                 dim_ff=cfg.convtran_dim_ff, 
                 dropout=cfg.convtran_dropout, 
                 num_classes=cfg.main_num_classes):
        super().__init__()

        if seq_len > 128:
            m = 2 ** int(math.log2(seq_len // 128))
        else:
            m = 1

        seq_len = seq_len // m

        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 15], padding='same'),
                                         nn.BatchNorm2d(emb_size*4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        self.maxpool = nn.MaxPool2d(kernel_size=(1, m), stride=(1, m))
        
        # Inception Feature Extractor -----------------------------------------------------------
        self.inception = InceptionTimeFeatureExtractor(
            n_in_channels=channel_size,
            out_channels=emb_size//4 # (out_channels*4=emb_size)
        )
        
        self.inc_proj = nn.Sequential(
            nn.MaxPool1d(kernel_size=m, stride=m),
            nn.Conv1d(emb_size, emb_size, kernel_size=1),
            nn.BatchNorm1d(emb_size),
            nn.GELU()
        )
        
        self.Fix_Position = tAPE(emb_size, dropout=dropout, max_len=seq_len)
        self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, dropout)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out1 = nn.Linear(emb_size, num_classes)
        self.out2 = nn.Linear(emb_size, 2)

    def forward(self, x, pad_mask=None):
        # input is (bs, 1, T, C)
        x = x.permute(0, 1, 3, 2) # (bs, 1, C, T)
        
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = self.maxpool(x_src) 
        
        x_inc = self.inception(x.squeeze(1)) # (bs, C, T)
        x_inc_proj = self.inc_proj(x_inc)
        
        x_src = x_src + x_inc_proj 
        
        x_src = x_src.permute(0, 2, 1)
        x_src_pos = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src_pos)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out1 = self.out1(out)
        out2 = self.out2(out)

        return out1, out2

class TimeCNN_SingleSensor_v1(nn.Module):
    """
    Based on "A deep learning architecture for temporal sleep stage classification
    using multivariate and multimodal time series"
    """

    def __init__(self, channel_size=cfg.imu_vars, T=cfg.seq_len, k=15, m=1, emb_size=128, num_heads=8, dim_ff=256, dropout=0.1, num_classes=cfg.main_num_classes):
        C = channel_size
        seq_len = T // m
        super().__init__()

        # Embedding Layer -----------------------------------------------------------
        self.Norm = nn.BatchNorm1d(C)
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(C, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(1, k))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=emb_size, out_channels=1,
                                           kernel_size=(1, k))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, m), stride=(1, m))

        # Attention Layer ------------------------------------------------------------
        self.Fix_Position = tAPE(emb_size, dropout=dropout, max_len=seq_len)
        self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, dropout)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout))
        self.out1 = nn.Linear(emb_size * seq_len, num_classes)
        self.out2 = nn.Linear(emb_size * seq_len, 2)

    def forward(self, x, pad_mask=None):
        # x: (bs, 1, T, C) -> (bs, 1, C, T)
        x = x.permute(0, 1, 3, 2)
        bs = x.shape[0]
        x = x.squeeze(1) # (bs, C, T)
        x = self.Norm(x) 
        x = x.unsqueeze(1) # (bs, 1, C, T)
        out = self.depthwise_conv(x).transpose(1, 2) # (bs, 1, emb_size, T)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv1(out)  # (bs, n_spatial_filters, C, T)
        out = self.relu(out)
        out = self.maxpool(out)  # (bs, n_spatial_filters, C, T // m)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)  # (bs, n_spatial_filters, C, T // m)
        out = self.relu(out)
        x_src = out.squeeze(1)  # (bs, embedding, T // m)
        x_src = x_src.transpose(1, 2)  # (bs, T // m, embedding)
        x_src_pos = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src_pos)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.view(bs, -1)  # (bs, n_spatial_filters * C * ((T // m) // m))
        out1 = self.out1(out)
        out2 = self.out2(out)
        return out1, out2