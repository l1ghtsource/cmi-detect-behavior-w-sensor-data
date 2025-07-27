import torch
from torch import nn
import torch.nn.functional as F

# take it from https://github.com/ZiyueWang25/Kaggle_G2Net/blob/main/1D_Model/src/models_1d.py

class GeM(nn.Module):
    '''
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    '''

    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        with torch.cuda.amp.autocast(enabled=False):  # to avoid NaN issue for fp16
            return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
               ', ' + 'eps=' + str(self.eps) + ')'

class Model1DCNNGEM(nn.Module):
    """1D convolutional neural network. Classifier of the gravitational waves.
    Architecture from there https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.141103
    """
    def __init__(self, channels=32, initial_channnels=8, num_classes=18):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(channels, initial_channnels, kernel_size=64),
            nn.BatchNorm1d(initial_channnels),
            nn.ELU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(initial_channnels, initial_channnels, kernel_size=32),
            GeM(kernel_size=8),
            nn.BatchNorm1d(initial_channnels),
            nn.ELU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(initial_channnels, initial_channnels * 2, kernel_size=32),
            nn.BatchNorm1d(initial_channnels * 2),
            nn.ELU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(initial_channnels * 2, initial_channnels * 2, kernel_size=16),
            GeM(kernel_size=6),
            nn.BatchNorm1d(initial_channnels * 2),
            nn.ELU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(initial_channnels * 2, initial_channnels * 4, kernel_size=16),
            nn.BatchNorm1d(initial_channnels * 4),
            nn.ELU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(initial_channnels * 4, initial_channnels * 4, kernel_size=16),
            GeM(kernel_size=4),
            nn.BatchNorm1d(initial_channnels * 4),
            nn.ELU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(initial_channnels * 4 * 11, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        # print(x.shape)
        x = x.flatten(1)
        # x = x.mean(-1)
        # x = torch.cat([x.mean(-1), x.max(-1)[0]])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x