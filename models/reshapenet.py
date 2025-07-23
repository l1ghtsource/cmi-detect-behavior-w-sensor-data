import torch.nn as nn
import timm
from configs.config import cfg

class ReshapeNet_SingleSensor_v1(nn.Module):
    def __init__(self, num_classes=cfg.main_num_classes):
        super().__init__()
        self.model = timm.create_model('timm/resnet18.a1_in1k', pretrained=True, in_chans=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(512, out_features=num_classes, bias=True)
        self.fc2 = nn.Linear(512, out_features=2, bias=True)
        self.fc3 = nn.Linear(512, out_features=4, bias=True)

    def extract_features(self, x):
        feature1 = self.model.forward_features(x)
        return feature1

    def forward(self, x, pad_mask=None):
        # x: (bs, n_sensors=1, seq_len=120, channels=32)
        bs = x.size(0)
        x = x.reshape(bs, 1, 60, 2, 32) # (bs, 1, 60, 2, 32)
        x = x.permute(0, 1, 2, 4, 3) # (bs, 1, 60, 32, 2)
        x = x.reshape(bs, 1, 60, 64) # (bs, 1, 60, 64)
        x = x.repeat(1, 3, 1, 1) # (bs, 3, 60, 64)
        x = self.extract_features(x)
        x = self.pool(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3