#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:36:52 2022

@author: Yangzhuobin

E-mail: yzb_98@tju.edu.cn

"""

import torch
import torch.nn as nn
import numpy as np

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=4, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class CNN(nn.Module):
    def __init__(self, F1: int = 64, D: int = 2):
        super(CNN, self).__init__()
        self.drop_out = 0.25
        self.att = cbam_block(D * F1)
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((7, 7, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 16),
                stride=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 8))
        )
        self.block_2 = nn.Sequential(
            nn.ZeroPad2d((7, 7, 0, 0)),
            nn.Conv2d(
                in_channels=F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),
            Swish(),#GLU(dim=1),)
            nn.Conv2d(
                in_channels= D * F1,
                out_channels=D * F1,
                kernel_size=(1, 16),
                stride=(1, 2),
                bias=False,
                groups=D * F1,
            ),
            nn.BatchNorm2d(D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(D * F1),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * D * F1, # D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False
            ),
            Swish(),#GLU(dim=1),)
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(3, 1),
                stride=(1, 1),
                groups=D * D * F1,
                bias=False
            ),
            nn.BatchNorm2d(D * D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * F1, # D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False
            ),
            nn.BatchNorm2d(D * F1), # D * D * F1
            # ChannelShuffle(D * D * F1, 4), # D * D * F1
        )
        self.block_4 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            nn.Conv2d(
                in_channels= D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),
            Swish(),
            nn.Conv2d(
                in_channels= D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 8),
                stride=(1, 1),
                bias=False,
                groups=D * D * F1,
            ),
            nn.BatchNorm2d(D * D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
                groups=4,
            ),
            nn.BatchNorm2d(D * F1),
            nn.AvgPool2d((2, 1)),
        )

    def forward(self, x, pad_mask=None):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.att(x)
        x = self.block_3(x)
        x = self.block_4(x)

        return x
    
class JNE_SingleSensor_v1(nn.Module):
    def __init__(self, F1: int = 64, D: int = 2, num_classes: int = 18,
                 dropout: float = 0.25):
        super().__init__()
        self.feature_extractor = CNN(F1=F1, D=D)
        self.pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.head1 = nn.Linear(D * F1, num_classes)
        self.head2 = nn.Linear(D * F1, 2)
        self.head3 = nn.Linear(D * F1, 4)

    def forward(self, x, pad_mask=None):
        x = self.feature_extractor(x)
        x = self.pre_head(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        return x1, x2, x3