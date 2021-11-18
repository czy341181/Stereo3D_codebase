import numpy as np
import torch
import torch.nn as nn
import math

class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernal_szie=3, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernal_szie,
                              stride=stride,
                              padding=kernal_szie//2,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    '''
    feat:
    torch.Size([16, 16, 384, 1280])   1
    torch.Size([16, 32, 192, 640])    1/2
    torch.Size([16, 64, 96, 320])     1/4
    torch.Size([16, 128, 48, 160])    1/8
    torch.Size([16, 256, 24, 80])     1/16
    torch.Size([16, 512, 12, 40])     1/32
    '''

class Adapt_Layer(nn.Module):
    def __init__(self, in_planes, src_ratio, affinity_flag=False):  #downsample_ratio:  1, 2, 4, 8
        ####下采样到1/32

        layers = []
        if affinity_flag == True:
            downsample_num = math.log2(32) - math.log2(src_ratio)

            for i in range(downsample_num):
                layers.append(Conv2d(in_planes, in_planes, kernal_szie=3, stride=2, bias=False))

        layers.append(Conv2d(in_planes, in_planes, kernal_szie=1, stride=1, bias=False))

        self.adapt_layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.adapt_layer(x)
        return x
