import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, kernel):
    return nn.Sequential(
        nn.Conv1d(in_ch, 128, kernel, padding="same"),
        nn.ReLU(inplace=True),
        nn.Conv1d(128, 64, kernel, padding="same"),
        nn.ReLU(inplace=True),
    )

class CNNLSTM(nn.Module):
    def __init__(self, kernels=(3, 5, 7), num_classes=18):
        super().__init__()
        self.kernels = kernels
        self.blocks = nn.ModuleDict()
        self._lstm = None
        self._fc   = None
        self.num_classes = num_classes

    def _lazy_init(self, in_ch):
        for k in self.kernels:
            self.blocks[str(k)] = conv_block(in_ch, k)

        lstm_in = 64 * len(self.kernels) + in_ch
        self._lstm = nn.LSTM(
            lstm_in, 128, num_layers=2,
            bidirectional=True, dropout=0.1, batch_first=True
        )
        self._fc = nn.Linear(128 * 2, self.num_classes)

    def forward(self, x):
        B, _, T, C = x.shape
        x = x.view(B, C, T)

        if not self.blocks:
            self._lazy_init(C)

        conv_out = [blk(x) for blk in self.blocks.values()]
        conv_out.append(x)
        concat = torch.cat(conv_out, dim=1)

        lstm_in = concat.transpose(1, 2)
        lstm_out, _ = self._lstm(lstm_in)

        pooled = lstm_out[:, -1]
        logits = self._fc(pooled)

        return logits