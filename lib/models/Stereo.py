import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from lib.losses.stereo_loss import compute_stereo_centernet3d_loss
#### CaDDN ####
#from lib.models.ffe.depth_ffe import DepthFFE
#from lib.models.f2v.frustum_to_voxel import FrustumToVoxel
#from lib.models.conv2d_collapse import Conv2DCollapse
#from lib.models.base_bev_backbone import BaseBEVBackbone
# from lib.models.dense_heads.anchor_head_single import AnchorHeadSingle
# from lib.models.depth_loss_head import DepthLossHead
#### Liga ####
# from lib.models.ligabackbone import LigaBackbone
# from lib.models.height_compression import HeightCompression
# from lib.models.hg_bev_backbone import HgBEVBackbone
# from lib.models.dense_heads.anchor_head_single import AnchorHeadSingle
# from lib.models.depth_loss_head import DepthLossHead

#### Stereo codebase
from lib.models.backbones import dla
from lib.models.backbones.dlaup import DLAUp
from lib.models.backbones import build_backbone_neck
from lib.models.detectionhead import DetHead

#from easydict import EasyDict as edict
#from lib.models.iou3d_nms import iou3d_nms_utils

class StereoNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, self.neck = build_backbone_neck(cfg)
        self.detection_head = DetHead(cfg)


        self.num_class = self.cfg['class_names']

    def forward(self, input, istrain=True):
        images_left = input['left_img']
        images_right = input['right_img']

        features_left = self.backbone(images_left)
        features_right = self.backbone(images_right)

        features_left = self.neck(features_left[2:])
        features_right = self.neck(features_right[2:])

        feature_concat = torch.cat((features_left, features_right), dim = 1)

        ret = self.detection_head(feature_concat)

        if istrain == True:
            rgb_loss, rgb_stats_batch = compute_stereo_centernet3d_loss(ret, input)
            return rgb_loss, rgb_stats_batch

        else:

            return ret



# if __name__ == '__main__':
#     import torch
#     net = CenterNet3D(backbone='dla34')
#     print(net)
#
#     input = torch.randn(4, 3, 384, 1280)
#     print(input.shape, input.dtype)
#     output = net(input)
#     print(output.keys())


