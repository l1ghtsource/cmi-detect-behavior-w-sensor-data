import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def torch_lineplot_gpu(imu_batch, img_h=1920, img_w=400, line_w=1.5):
    bs, _, T, C = imu_batch.shape
    dev = imu_batch.device

    sig = imu_batch.transpose(2, 3).squeeze(1)
    sig_min = sig.amin(dim=2, keepdim=True)
    sig_max = sig.amax(dim=2, keepdim=True)
    sig = (sig - sig_min) / (sig_max - sig_min + 1e-6)
    row_h = img_h // C

    t = torch.arange(T, device=dev).float() / (T - 1) * (img_w - 1)
    t = t.view(1, 1, -1).expand(bs, C, -1)
    y = sig * (row_h - 1)
    y = y + torch.arange(C, device=dev).view(1, -1, 1) * row_h

    x0 = t.floor().long()
    x1 = (x0 + 1).clamp(max=img_w - 1)
    y0 = y.floor().long()
    y1 = (y0 + 1).clamp(max=img_h - 1)

    wx1 = (t - x0.float())
    wy1 = (y - y0.float())
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    canvas = torch.zeros(bs, 1, img_h, img_w, device=dev)

    for (ix, iy, w) in [
        (x0, y0, wx0 * wy0),
        (x1, y0, wx1 * wy0),
        (x0, y1, wx0 * wy1),
        (x1, y1, wx1 * wy1),
    ]:
        idx = iy * img_w + ix
        canvas.reshape(bs, 1, -1).scatter_add_(
            2,
            idx.reshape(bs, 1, -1),
            w.reshape(bs, 1, -1)
        )

    sigma = line_w / 2.3548
    k = max(int(3 * sigma) * 2 + 1, 3)
    g = torch.arange(k, device=dev) - k // 2
    gauss = torch.exp(-(g**2)/(2*sigma**2))
    gauss = gauss / gauss.sum()
    gauss2d = gauss.view(1, 1, 1, -1) * gauss.view(1, 1, -1, 1)
    canvas = F.conv2d(canvas, gauss2d, padding=k//2, groups=1)

    canvas = canvas.expand(-1, 3, -1, -1)
    canvas = canvas.clamp(0, 1)
    
    return canvas

class SignalModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout=0.3):
        super().__init__()

        self.backbone = timm.create_model(
            'timm/convnextv2_nano.fcmae_ft_in22k_in1k',
            pretrained=pretrained,
            dropout_rate=dropout,
            num_classes=0
        )
        feat_dim = 640

        self.register_buffer('norm_mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('norm_std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self.head1 = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )
        self.head2 = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2)
        )
        self.head3 = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 4)
        )

    def forward(self, x, pad_mask=None):
        x = torch_lineplot_gpu(x, img_h=1920, img_w=400)
        x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
        x = (x - self.norm_mean) / self.norm_std
        feats = self.backbone(x)
        logits1 = self.head1(feats)
        logits2 = self.head2(feats)
        logits3 = self.head3(feats)
        return logits1, logits2, logits3