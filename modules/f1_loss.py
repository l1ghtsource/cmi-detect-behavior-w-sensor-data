import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import cfg

class F1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        y_true_one_hot = F.one_hot(y_true, num_classes=cfg.num_classes).float()

        tp = (y_true_one_hot * y_pred_softmax).sum(dim=0)
        fp = ((1 - y_true_one_hot) * y_pred_softmax).sum(dim=0)
        fn = (y_true_one_hot * (1 - y_pred_softmax)).sum(dim=0)

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)

        return 1 - f1.clamp(min=self.eps, max=1-self.eps).mean()