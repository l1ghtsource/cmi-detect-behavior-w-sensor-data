import torch
import torch.nn as nn

class Wave_Block(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dilation_rates, kernel_size):
        super().__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs   = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels,
                                    out_channels, kernel_size=1))

        dilations = [2 ** i for i in range(dilation_rates)]
        for d in dilations:
            pad = (d * (kernel_size - 1)) // 2
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels,
                          kernel_size, padding=pad, dilation=d))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels,
                          kernel_size, padding=pad, dilation=d))
            self.convs.append(nn.Conv1d(out_channels,
                                        out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * \
                torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res

class Classifier(nn.Module):
    def __init__(self, in_ch=3, kernel=3, num_classes=3):
        super().__init__()

        self.block2 = Wave_Block(in_ch, 32, 8, kernel)
        self.block3 = Wave_Block(32, 64, 4, kernel)
        self.block4 = Wave_Block(64, 128, 1, kernel)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.permute(0, 2, 1)
        seq_out, _ = self.gru(x)

        last_step = seq_out[:, -1, :]
        logits = self.fc(last_step)
        
        return logits