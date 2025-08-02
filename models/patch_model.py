import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    '''
    (bs,1,120,32) -> (bs, N_patches=10, patch_size*channels=384)
    '''
    def __init__(self, patch_size:int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        bs, _, seq_len, ch = x.shape
        assert seq_len % self.patch_size == 0, 'seq_len % patch_size != 0'
        n_patches = seq_len // self.patch_size
        # reshape to patches
        x = x.view(bs, n_patches, self.patch_size, ch) # (bs, 10, 12, 32)
        x = x.flatten(2) # (bs, 10, 384)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len:int, dim:int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, seq_len, dim))

    def forward(self, x):
        return x + self.pos

class EncoderBlock(nn.Module):
    def __init__(self, dim:int, heads:int, dropout:float):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x): # (bs, seq, dim)
        attn_out, _ = self.mha(x, x, x) # self-attention
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class Encoder(nn.Module):
    def __init__(self, patch_size, model_dim, dropout, num_heads, num_enc_layers, num_lstm_layers):
        super().__init__()
        self.patch = PatchEmbed(patch_size)
        n_patches = 120 // patch_size

        self.input_linear = nn.Linear(patch_size * 32, model_dim)
        self.pos_enc = PositionalEncoding(n_patches, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.encoders = nn.ModuleList([
            EncoderBlock(model_dim,
                         num_heads,
                         dropout) for _ in range(num_enc_layers)
        ])

        self.lstms = nn.ModuleList([
            nn.LSTM(model_dim,
                    model_dim,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True) for _ in range(num_lstm_layers)
        ])

    def forward(self, x):
        x = self.patch(x)
        x = self.input_linear(x)

        x = self.pos_enc(x)
        x = self.dropout(x)

        for enc in self.encoders:
            x = enc(x)

        for lstm in self.lstms:
            x, _ = lstm(x)

        return x

class PatchModel(nn.Module):
    def __init__(
            self, 
            patch_size=4, 
            model_dim=64, 
            dropout=0.1, 
            num_heads=8, 
            num_enc_layers=6,
            num_lstm_layers=1, 
            num_classes=18
        ):
        super().__init__()
        self.encoder = Encoder(patch_size, model_dim, dropout, num_heads, num_enc_layers, num_lstm_layers)
        self.head1 = nn.Linear(model_dim * 2, num_classes)
        self.head2 = nn.Linear(model_dim * 2, 2)
        self.head3 = nn.Linear(model_dim * 2, 4)

    def forward(self, x, pad_mask=None):
        x = self.encoder(x)
        x = x.mean(dim=1)
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        return x1, x2, x3