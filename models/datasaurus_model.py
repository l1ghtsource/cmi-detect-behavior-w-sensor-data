import timm
import torch
import torch.nn as nn
from nnAudio import Spectrogram

# take it from https://github.com/ZiyueWang25/Kaggle_G2Net/blob/main/datasaurus/src/tcnn.py

def nanstd_mean(v, *args, inplace=False, unbiased=True, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    is_inf = torch.isinf(v)
    v[is_nan] = 0
    v[is_inf] = 0

    mean = v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
    numerator = ((v - mean) ** 2).sum(*args, **kwargs)
    N = (~is_nan).float().sum(*args, **kwargs)

    if unbiased:
        N -= 1

    std = torch.sqrt(numerator / N)
    return std, mean

def imagenet_norm(x):
    means = torch.tensor([0.485, 0.456, 0.406], device=x.device)
    stds = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    means = torch.broadcast_to(means, (x.shape[0], 3)).unsqueeze(-1).unsqueeze(-1)
    stds = torch.broadcast_to(stds, (x.shape[0], 3)).unsqueeze(-1).unsqueeze(-1)
    return x * stds + means

def standard_scaler(features, imagenet=False):
    std, mean = nanstd_mean(features, dim=[2, 3], keepdim=True)
    features = (features - mean) / std
    features = torch.nan_to_num(features, 0, 5, -5)

    if imagenet:
        return imagenet_norm(features)
    else:
        return features

class GeM(nn.Module):
    """
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    """
    def __init__(self, kernel_size=8, p=3, eps=1e-6, adaptive=False):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps
        self.adaptive = adaptive

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        with torch.cuda.amp.autocast(enabled=False):
            x_clamped = x.clamp(min=eps).pow(p)
            
            if self.adaptive or x_clamped.size(-1) <= self.kernel_size:
                output_size = max(1, x_clamped.size(-1) // self.kernel_size)
                return nn.functional.adaptive_avg_pool1d(x_clamped, output_size).pow(1.0 / p)
            else:
                return nn.functional.avg_pool1d(x_clamped, self.kernel_size).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )

class ResBlockGeM(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        downsample=1,
        act=nn.SiLU(inplace=True),
    ):
        super().__init__()
        self.act = act
        if downsample != 1 or in_channels != out_channels:
            self.residual_function = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                act,
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                GeM(kernel_size=downsample),  # downsampling
            )
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                GeM(kernel_size=downsample),  # downsampling
            )  # skip layers in residual_function, can try simple MaxPool1d
        else:
            self.residual_function = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                act,
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"
    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class ModelIafossV2(nn.Module):
    def __init__(
        self,
        n: int = 32,
        nh: int = 512,
        num_classes: int = 18,
        in_channels: int = 32,
        act: nn.Module = nn.SiLU(inplace=True),
        ps: float = 0.3,
    ):
        super().__init__()
        self.n = n

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=n,
                      kernel_size=15, padding=15//2, bias=False),
            nn.BatchNorm1d(n),
            act,
            nn.Conv1d(n, n, kernel_size=15, padding=15//2, bias=False),
            GeM(kernel_size=2),
            ResBlockGeM(n, n, kernel_size=15, downsample=2, act=act),
            ResBlockGeM(n, n, kernel_size=15,               act=act),
        )

        self.conv1 = nn.Sequential(
            ResBlockGeM(n,   2*n, kernel_size=15, downsample=2, act=act),
            ResBlockGeM(2*n, 2*n, kernel_size=15, act=act),
            ResBlockGeM(2*n, 4*n, kernel_size=15, downsample=2, act=act),
            ResBlockGeM(4*n, 4*n, kernel_size=15, act=act),
        )

        self.conv2 = nn.Sequential(
            ResBlockGeM(4*n, 8*n, kernel_size=7, downsample=2, act=act),
            ResBlockGeM(8*n, 8*n, kernel_size=7, act=act), 
            ResBlockGeM(8*n, 8*n, kernel_size=7, act=act),
        )

        self.head1 = nn.Sequential(
            AdaptiveConcatPool1d(),
            nn.Flatten(),
            nn.Linear(8*n*2, nh),
            nn.BatchNorm1d(nh), nn.Dropout(ps), act,
            nn.Linear(nh, nh // 2),
            nn.BatchNorm1d(nh // 2), nn.Dropout(ps), act,
            nn.Linear(nh // 2, num_classes),
        )
        self.head2 = nn.Sequential(
            AdaptiveConcatPool1d(),
            nn.Flatten(),
            nn.Linear(8*n*2, nh),
            nn.BatchNorm1d(nh), nn.Dropout(ps), act,
            nn.Linear(nh, nh // 2),
            nn.BatchNorm1d(nh // 2), nn.Dropout(ps), act,
            nn.Linear(nh // 2, 2),
        )
        self.head3 = nn.Sequential(
            AdaptiveConcatPool1d(),
            nn.Flatten(),
            nn.Linear(8*n*2, nh),
            nn.BatchNorm1d(nh), nn.Dropout(ps), act,
            nn.Linear(nh, nh // 2),
            nn.BatchNorm1d(nh // 2), nn.Dropout(ps), act,
            nn.Linear(nh // 2, 4),
        )

    def forward(self, x, pad_mask=None):
        x = x.squeeze(1).permute(0, 2, 1)
        out = self.stem(x)
        out = self.conv1(out)
        features = self.conv2(out)
        out1 = self.head1(features)
        out2 = self.head2(features)
        out3 = self.head3(features)
        return out1, out2, out3

class CQT2D(nn.Module):
    def __init__(
        self,
        encoder: str = "resnet34",
        num_classes: int = 18,
        sample_rate: int = 10,
        hop_length: int = 4,
        n_cqt: int = 16,
    ):
        super().__init__()
        self.n_cqt = n_cqt

        self.model2d = timm.create_model(
            encoder,
            pretrained=False,
            num_classes=0,
            in_chans=n_cqt + 1,
        )

        with torch.no_grad():
            dummy = torch.randn(1, n_cqt + 1, 224, 224)
            emb_2d = self.model2d(dummy).shape[-1]

        self.head1 = nn.Sequential(
            nn.Linear(emb_2d, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        self.head2 = nn.Sequential(
            nn.Linear(emb_2d, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        self.head3 = nn.Sequential(
            nn.Linear(emb_2d, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )

        self.spec_transform = Spectrogram.CQT1992v2(
            sr=sample_rate,
            fmin=0.1,
            fmax=5.0,
            hop_length=hop_length,
            window="flattop",
            bins_per_octave=24,
            filter_scale=0.5,
        )

    @staticmethod
    def _add_frequency_encoding(spec: torch.Tensor) -> torch.Tensor:
        B, C, F, T = spec.shape
        freq_enc = 2 * torch.arange(F, device=spec.device) / F - 1
        freq_enc = freq_enc[None, None, :, None].repeat(B, 1, 1, T)
        return torch.cat([spec, freq_enc], dim=1)

    def _prepare_image(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, C_src = x.shape
        x = x.squeeze(1).permute(0, 2, 1)
        spec = self.spec_transform(x.reshape(-1, T))
        F, T2 = spec.shape[-2], spec.shape[-1]
        spec = spec.reshape(B, C_src, F, T2)

        if C_src >= self.n_cqt:
            spec = spec[:, :self.n_cqt]
        else:
            pad = self.n_cqt - C_src
            spec = torch.cat([spec,
                              spec[:, -1:].repeat(1, pad, 1, 1)], dim=1)

        spec = standard_scaler(spec)
        spec = self._add_frequency_encoding(spec)
        return spec

    def forward(self, x, pad_mask=None) -> torch.Tensor:
        img = self._prepare_image(x)
        feats = self.model2d(img)
        out1 = self.head1(feats)
        out2 = self.head2(feats)
        out3 = self.head3(feats)
        return out1, out2, out3