import torch
import torch.nn as nn


class DetHead(nn.Module):
    def __init__(self, cfg):
        super(DetHead, self).__init__()
        channels = cfg['detection_head']['input_feature']
        head_conv = cfg['detection_head']['head_conv']
        self.heatmap = nn.Sequential(nn.Conv2d(channels, head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(head_conv, 3, kernel_size=1, stride=1, padding=0, bias=True))
        self.offset_2d_left = nn.Sequential(nn.Conv2d(channels, head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.size_2d_left = nn.Sequential(nn.Conv2d(channels, head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))

        self.offset_2d_right = nn.Sequential(nn.Conv2d(channels, head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))

        self.width_right = nn.Sequential(nn.Conv2d(channels, head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))

        self.heatmap[-1].bias.data.fill_(-2.19)
        self.fill_fc_weights(self.offset_2d_left)
        self.fill_fc_weights(self.size_2d_left)
        self.fill_fc_weights(self.offset_2d_right)
        self.fill_fc_weights(self.width_right)

    def forward(self, feat):
        ret = {}
        
        ret['heatmap'] = self.heatmap(feat)
        ret['offset_2d_left'] = self.offset_2d_left(feat)
        ret['size_2d_left'] = self.size_2d_left(feat)
        ret['offset_2d_right'] = self.offset_2d_right(feat)
        ret['width_right'] = self.width_right(feat)

        return ret

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)