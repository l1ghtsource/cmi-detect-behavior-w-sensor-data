import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import cfg

# bad solo, bad in hybrid :(

class Wave_Block(nn.Module):
    def __init__(self, in_ch, out_ch, n_rates, k):
        super().__init__()
        self.conv_in = nn.Conv1d(in_ch, out_ch, 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(out_ch, out_ch * 2, k,
                      padding=((k - 1) * (2 ** i)) // 2,
                      dilation=2 ** i)
            for i in range(n_rates)
        ])
        self.conv_out = nn.Conv1d(out_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv_in(x)
        res = x
        for c in self.convs:
            f, g = c(x).chunk(2, 1)
            x = torch.tanh(f) * torch.sigmoid(g)
        return res + self.conv_out(x)


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(SEModule, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        s = F.adaptive_avg_pool1d(x, 1)
        s = self.conv(s)
        x *= torch.sigmoid(s)
        return x

class WaveNet_SingleSensor_v1(nn.Module):
    def __init__(self, kernel_size=3, num_classes=cfg.main_num_classes):
        super().__init__()
        dropout = 0.1
        inch = cfg.imu_vars

        self.conv1d_in = nn.Sequential(
            nn.Conv1d(inch, 32, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout)
        )

        self.block1 = Wave_Block(32, 16, 6,  kernel_size)
        self.block2 = Wave_Block(inch+16, 32, 4, kernel_size)
        self.block3 = Wave_Block(inch+16+32, 64, 2, kernel_size)
        self.block4 = Wave_Block(inch+16+32+64, 128, 1, kernel_size)

        self.se1, self.se2, self.se3, self.se4 = SEModule(16), SEModule(32), SEModule(64), SEModule(128)
        self.bn1, self.bn2, self.bn3, self.bn4 = nn.BatchNorm1d(16), nn.BatchNorm1d(32), nn.BatchNorm1d(64), nn.BatchNorm1d(128)
        self.dp1, self.dp2, self.dp3, self.dp4 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

        enc_in_dim = inch + 16 + 32 + 64 + 128
        self.lstm = nn.LSTM(enc_in_dim, 64, num_layers=1, batch_first=True, bidirectional=True)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.class_head1 = nn.Linear(128, num_classes)
        self.class_head2 = nn.Linear(128, 2)

    def forward(self, imu_data, pad_mask=None):
        x = imu_data.squeeze(1).permute(0, 2, 1)      # [B, C, L]

        x0 = self.conv1d_in(x)

        x1 = self.dp1(self.bn1(self.se1(self.block1(x0))))
        x2_base = torch.cat([x1, x], dim=1)

        x2 = self.dp2(self.bn2(self.se2(self.block2(x2_base))))
        x3_base = torch.cat([x2_base, x2], dim=1)

        x3 = self.dp3(self.bn3(self.se3(self.block3(x3_base))))
        x4_base = torch.cat([x3_base, x3], dim=1)

        x4 = self.dp4(self.bn4(self.se4(self.block4(x4_base))))
        x5_base = torch.cat([x4_base, x4], dim=1)

        lstm_in = x5_base.permute(0, 2, 1)            # [B, L, C]
        lstm_out, _ = self.lstm(lstm_in)              # [B, L, 128]

        pooled = self.pooling(lstm_out.permute(0, 2, 1)).squeeze(-1)  # [B, 128]

        logits1 = self.class_head1(pooled)              # [B, num_classes]
        logits2 = self.class_head2(pooled)              # [B, 2]

        return logits1, logits2