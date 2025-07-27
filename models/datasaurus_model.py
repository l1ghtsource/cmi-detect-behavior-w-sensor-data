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
    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        with torch.cuda.amp.autocast(enabled=False):  # to avoid NaN issue for fp16
            return nn.functional.avg_pool1d(
                x.clamp(min=eps).pow(p), self.kernel_size
            ).pow(1.0 / p)

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

class Extractor(nn.Sequential):
    def __init__(
        self, in_c=8, out_c=8, kernel_size=64, maxpool=8, act=nn.SiLU(inplace=True)
    ):
        super().__init__(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_c),
            act,
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2),
            # nn.MaxPool1d(kernel_size=maxpool),
            GeM(kernel_size=maxpool),
        )

class ModelIafossV2(nn.Module):
    def __init__(
        self,
        n: int = 8,
        nh: int = 256,
        num_classes: int = 1,
        act: nn.Module = nn.SiLU(inplace=True),
        ps: float = 0.5,
    ):
        super().__init__()
        self.n = n

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=None,
                      out_channels=n,
                      kernel_size=127, padding=127//2, bias=False),
            nn.BatchNorm1d(n),
            act,
            nn.Conv1d(n, n, kernel_size=127, padding=127//2, bias=False),
            GeM(kernel_size=2),
            ResBlockGeM(n, n, kernel_size=31, downsample=4, act=act),
            ResBlockGeM(n, n, kernel_size=31,               act=act),
        )

        self.conv1 = nn.Sequential(
            ResBlockGeM(n,   n,   kernel_size=31, downsample=4, act=act),
            ResBlockGeM(n,   n,   kernel_size=31,               act=act),
            ResBlockGeM(n, 3*n,  kernel_size=31, downsample=4, act=act),
            ResBlockGeM(3*n, 3*n, kernel_size=31,               act=act),
        )

        self.conv2 = nn.Sequential(
            ResBlockGeM(6*n, 4*n, kernel_size=15, downsample=4, act=act),
            ResBlockGeM(4*n, 4*n, kernel_size=15,               act=act),
            ResBlockGeM(4*n, 8*n, kernel_size=7,  downsample=4, act=act),
            ResBlockGeM(8*n, 8*n, kernel_size=7,                act=act),
        )

        self.head = nn.Sequential(
            AdaptiveConcatPool1d(),
            nn.Flatten(),
            nn.Linear(8*n*2, nh),
            nn.BatchNorm1d(nh), nn.Dropout(ps), act,
            nn.Linear(nh, nh),
            nn.BatchNorm1d(nh), nn.Dropout(ps), act,
            nn.Linear(nh, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = x.squeeze(1).permute(0, 2, 1)
        C_in = x.size(1)

        if isinstance(self.stem[0], nn.Conv1d) and self.stem[0].in_channels is None:
            self.stem[0] = nn.Conv1d(C_in, self.n,
                                     kernel_size=127, padding=127//2, bias=False).to(x.device)

        out = self.stem(x)
        x1a = self.conv1[:2](out)
        x1b = self.conv1[2:](torch.cat([out, out, out], dim=1))
        x1 = torch.cat([x1a, x1b], dim=1)
        features = self.conv2(x1)
        return self.head(features)

class CQT2D(nn.Module):
    def __init__(
        self,
        encoder: str = "resnet18",
        num_classes: int = 18,
        sample_rate: int = 10,
        hop_length: int = 8,
        n_cqt: int = 3,
    ):
        super().__init__()
        self.n_cqt = n_cqt

        self.model2d = timm.create_model(
            encoder,
            pretrained=True,
            num_classes=0,
            in_chans=n_cqt + 1,
        )

        with torch.no_grad():
            dummy = torch.randn(1, n_cqt + 1, 224, 224)
            emb_2d = self.model2d(dummy).shape[-1]

        self.head = nn.Sequential(
            nn.Linear(emb_2d, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self.spec_transform = Spectrogram.CQT1992v2(
            sr=sample_rate,
            fmin=20,
            fmax=1_000,
            hop_length=hop_length,
            window="flattop",
            bins_per_octave=48,
            filter_scale=0.25,
        )

    @staticmethod
    def _add_frequency_encoding(spec: torch.Tensor) -> torch.Tensor:
        B, C, F, T = spec.shape
        freq_enc = 2 * torch.arange(F, device=spec.device) / F - 1
        freq_enc = freq_enc[None, None, :, None].repeat(B, 1, 1, T)
        return torch.cat([spec, freq_enc], dim=1)

    def _prepare_image(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, C_src = x.shape
        x = x.squeeze(1).permute(0, 3, 2)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = self._prepare_image(x)
        feats = self.model2d(img)
        return self.head(feats)