import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
from lib.losses.dim_aware_loss import dim_aware_l1_loss


def compute_stereo_centernet3d_loss(input, target):
    stats_dict = {}

    seg_loss = compute_segmentation_loss(input, target)
    offset2d_left_loss = compute_offset_2d_left_loss(input, target)
    offset2d_right_loss = compute_offset_2d_right_loss(input, target)
    size_2d_left_loss = compute_size_2d_left_loss(input, target)
    width_right_loss = compute_width_right_loss(input, target)

    # statistics
    stats_dict['seg_loss'] = seg_loss.item()
    stats_dict['offset2d_left_loss'] = offset2d_left_loss.item()
    stats_dict['offset2d_right_loss'] = offset2d_right_loss.item()
    stats_dict['size_2d_left_loss'] = size_2d_left_loss.item()
    stats_dict['width_right_loss'] = width_right_loss.item()

    total_loss = seg_loss + offset2d_left_loss + offset2d_right_loss + size_2d_left_loss + width_right_loss
    return total_loss, stats_dict


def compute_segmentation_loss(input, target):
    input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
    loss = focal_loss_cornernet(input['heatmap'], target['heatmap'])
    return loss


def compute_size_2d_left_loss(input, target):
    # compute size2d loss
    size_2d_left_input = extract_input_from_tensor(input['size_2d_left'], target['indices'], target['mask_2d'])
    size_2d_left_target = extract_target_from_tensor(target['size_2d_left'], target['mask_2d'])
    size_2d_left_loss = F.l1_loss(size_2d_left_input, size_2d_left_target, reduction='mean')
    return size_2d_left_loss

def compute_width_right_loss(input, target):
    # compute size2d loss
    width_right_input = extract_input_from_tensor(input['width_right'], target['indices'], target['mask_2d'])
    width_right_target = extract_target_from_tensor(target['width_right'], target['mask_2d'])
    width_right_loss = F.l1_loss(width_right_input, width_right_target, reduction='mean')
    return width_right_loss

def compute_offset_2d_left_loss(input, target):
    # compute offset2d loss
    offset2d_input = extract_input_from_tensor(input['offset_2d_left'], target['indices'], target['mask_2d'])
    offset2d_target = extract_target_from_tensor(target['offset_2d_left'], target['mask_2d'])
    offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')
    return offset2d_loss

def compute_offset_2d_right_loss(input, target):
    # compute offset2d loss
    offset2d_input = extract_input_from_tensor(input['offset_2d_right'], target['indices'], target['mask_2d'])
    offset2d_target = extract_target_from_tensor(target['offset_2d_right'], target['mask_2d'])
    offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')
    return offset2d_loss


def compute_depth_loss(input, target):
    depth_input = extract_input_from_tensor(input['depth'], target['indices'], target['mask_3d'])
    depth_input, depth_log_variance = depth_input[:, 0:1], depth_input[:, 1:2]
    depth_input = 1. / (depth_input.sigmoid() + 1e-6) - 1.
    depth_target = extract_target_from_tensor(target['depth'], target['mask_3d'])
    depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, depth_target, depth_log_variance)
    return depth_loss


def compute_offset3d_loss(input, target):
    offset3d_input = extract_input_from_tensor(input['offset_3d'], target['indices'], target['mask_3d'])
    offset3d_target = extract_target_from_tensor(target['offset_3d'], target['mask_3d'])
    offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')
    return offset3d_loss


def compute_size3d_loss(input, target):
    size3d_input = extract_input_from_tensor(input['size_3d'], target['indices'], target['mask_3d'])
    size3d_target = extract_target_from_tensor(target['size_3d'], target['mask_3d'])
    size3d_loss = dim_aware_l1_loss(size3d_input, size3d_target, size3d_input)
    return size3d_loss


def compute_heading_loss(input, target):
    heading_input = _transpose_and_gather_feat(input['heading'], target['indices'])   # B * C * H * W ---> B * K * C
    heading_input = heading_input.view(-1, 24)
    heading_target_cls = target['heading_bin'].view(-1)
    heading_target_res = target['heading_res'].view(-1)
    mask = target['mask_2d'].view(-1)

    # classification loss
    heading_input_cls = heading_input[:, 0:12]
    heading_input_cls, heading_target_cls = heading_input_cls[mask], heading_target_cls[mask]
    if mask.sum() > 0:
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='mean')
    else:
        cls_loss = 0.0

    # regression loss
    heading_input_res = heading_input[:, 12:24]
    heading_input_res, heading_target_res = heading_input_res[mask], heading_target_res[mask]
    cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
    heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
    reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='mean')
    return cls_loss + reg_loss


######################  auxiliary functions #########################




def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind.long())  # B*C*H*W --> B*K*C
    return input[mask.long()]  # B*K*C --> M * C

def extract_target_from_tensor(target, mask):
    return target[mask.long()]


if __name__ == '__main__':
    input_cls  = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg  = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))

