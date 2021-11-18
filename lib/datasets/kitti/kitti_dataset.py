import os
import numpy as np
import torch.utils.data as data
from PIL import Image

import skimage.transform
from collections import defaultdict
import copy

from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import boxes3d_kitti_camera_to_lidar, boxes3d_kitti_camera_to_pselidar, \
    boxes3d_lidar_to_kitti_camera, boxes3d_kitti_camera_to_imageboxes
from lib.datasets.kitti.kitti_utils import random_flip
from lib.datasets.kitti.kitti_utils import mask_boxes_outside_range_numpy
from lib.datasets.kitti.kitti_utils import get_pad_params
from lib.datasets.kitti.kitti_utils import limit_period
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
from lib.datasets.kitti.kitti_utils import boxes3d_pselidar_to_kitti_camera
from lib.datasets.kitti.kitti_utils import points_to_depth_map
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result

from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian

import cv2

#from lib.models.roiaware_pool3d import roiaware_pool3d_utils

import scipy.misc


class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):
        # basic configuration
        self.max_objs = 50
        self.resolution = np.array(cfg['crop_size'])
        self.root_dir = cfg.get('root_dir', '../../data/KITTI')
        self.split = split
        self.num_classes = len(cfg['writelist'])
        self.class_name = cfg['writelist']
        self.cls2id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Van': 3, 'Truck': 4}

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'object', 'testing' if split == 'test' else 'training')
        # self.image_dir = os.path.join(self.data_dir, 'image_2')
        # self.image_dir = os.path.join(self.data_dir, 'image_2')
        #self.depth_dir = os.path.join(self.data_dir, 'depth_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        self.label_dir_3 = os.path.join(self.data_dir, 'label_3')
        # data augmentation configuration
        self.istrain = True if split in ['train', 'trainval'] else False

        # monocular
        self.downsample = 4
        #self.depth_downsample_factor = 4
        #self.point_cloud_range = cfg['pc_range']

        # stereo
        #self.use_van = True
        #self.use_person_sitting = True
        self.flip_type = cfg['flip_type']
        self.crop_size = cfg['crop_size']
        #self.gen_depth = cfg['gen_depth']

    def get_image(self, idx, image_id=2):
        img_file = os.path.join(self.data_dir + ('/image_%s' % image_id), '%06d.png' % idx)
        assert os.path.exists(img_file)
        image = Image.open(img_file).convert("RGB")
        return image


    def get_left_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_right_label(self, idx):
        label_file = os.path.join(self.label_dir.replace('label_2','label_3'), '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)


    def eval(self, results_dir, logger, label_flag):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        if label_flag == 'left':
            gt_annos = kitti.get_label_annos(self.label_dir, img_ids)
        elif label_flag == 'right':
            gt_annos = kitti.get_label_annos(self.label_dir_3, img_ids)

        test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

        logger.info('==> Evaluating (official) ...')
        for category in self.class_name:
            results_str, results_dict = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            logger.info(results_str)

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        input = {}
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id

        img_l = self.get_image(index, image_id=2)
        img_r = self.get_image(index, image_id=3)

        # if self.istrain==True:
        #     depth = self.get_depth_map(index)
        img_size = np.array(img_l.size[:2], dtype=np.int32)
        features_size = self.resolution // self.downsample    # W * H

        calib = self.get_calib(index)
        calib_ori = copy.deepcopy(calib)
        if self.istrain == True:
            l_objects = self.get_left_label(index)
            r_objects = self.get_right_label(index)

            gt_names = []
            gt_truncated = []
            gt_occluded = []

            locations = []
            left_project_center = []
            right_project_center = []
            dims = []
            rotations_y = []
            left_bbox = []
            right_bbox = []

            for l_object, r_object in zip(l_objects, r_objects):
                # if self.use_van:  # default: True
                #     # Car 14357, Van 1297
                #     if l_object.cls_type == 'Van':
                #         l_object.cls_type = 'Car'
                #
                # if self.use_person_sitting:  # default: True
                #     # Ped 2207, Person_sitting 56
                #     if l_object.cls_type == 'Person_sitting':
                #         l_object.cls_type = 'Pedestrian'
                if l_object.cls_type not in self.class_name + ['Van', 'Truck']:  #DontCare
                    continue

                if float(l_object.trucation) >= 0.98:
                    continue

                center_3d = l_object.pos + [0, -l_object.h / 2, 0]
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d_l, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d_l = center_3d_l[0]  # shape adjustment

                center_3d_r, _ = calib.rect_to_rightimg(center_3d)  # project 3D center to image plane
                center_3d_r = center_3d_r[0]  # shape adjustment


                gt_names.append(l_object.cls_type)
                locations.append(l_object.pos)
                left_project_center.append(center_3d_l)
                right_project_center.append(center_3d_r)
                dims.append([l_object.l, l_object.h, l_object.w])
                rotations_y.append(l_object.ry)
                left_bbox.append(l_object.box2d)
                right_bbox.append(r_object.box2d)
                gt_truncated.append(l_object.trucation)
                gt_occluded.append(l_object.occlusion)

            gt_names = np.array(gt_names)
            locations = np.array(locations)
            left_project_center = np.array(left_project_center)
            right_project_center = np.array(right_project_center)
            dims = np.array(dims)
            rotations_y = np.array(rotations_y)
            left_bbox = np.array(left_bbox)
            right_bbox = np.array(right_bbox)
            gt_truncated = np.array(gt_truncated)
            gt_occluded = np.array(gt_occluded)
            gt_classes = np.array([self.cls2id[n] for n in gt_names], dtype=np.int32)

        if self.istrain == True:
            img_l, img_r, left_bbox, right_bbox, left_project_center, right_project_center, calib, flip_this_image = \
                random_flip(img_l, img_r, left_bbox, right_bbox, left_project_center, right_project_center, calib)

            union_bbox = copy.copy(left_bbox)
            for i in range(union_bbox.shape[0]):
                union_bbox[i, 0] = min(left_bbox[i, 0], right_bbox[i, 0])
                union_bbox[i, 1] = min(left_bbox[i, 1], right_bbox[i, 1])
                union_bbox[i, 2] = max(left_bbox[i, 2], right_bbox[i, 2])
                union_bbox[i, 3] = max(left_bbox[i, 3], right_bbox[i, 3])



            heatmap = np.zeros((self.num_classes, self.resolution[0]//4, self.resolution[1]//4), dtype=np.float32)  # C * H * W
            offset_2d_left = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d_right = np.zeros((self.max_objs, 2), dtype=np.float32)
            size_2d_left = np.zeros((self.max_objs, 2), dtype=np.float32)
            width_right = np.zeros((self.max_objs, 1), dtype=np.float32)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.uint8)

            left_project_center_copy = copy.copy(left_project_center)
            right_project_center_copy = copy.copy(right_project_center)

            left_bbox_copy = copy.copy(left_bbox)
            right_bbox_copy = copy.copy(right_bbox)

            for i in range(union_bbox.shape[0]):
                left_box = left_bbox_copy[i, :]
                right_box = right_bbox_copy[i, :]
                left_box /= self.downsample
                right_box /= self.downsample

                left_project_center_item = left_project_center_copy[i, :]
                right_project_center_item = right_project_center_copy[i, :]
                left_project_center_item /= self.downsample
                right_project_center_item /= self.downsample

                center_2d_left = np.array([(left_box[0] + left_box[2]) / 2, (left_box[1] + left_box[3]) / 2],
                                     dtype=np.float32)  # W * H
                center_2d_right = np.array([(right_box[0] + right_box[2]) / 2, (right_box[1] + right_box[3]) / 2],
                                     dtype=np.float32)  # W * H

                left_project_center_item = left_project_center_item.astype(np.int32)
                right_project_center_item = right_project_center_item.astype(np.int32)

                if left_project_center_item[0] < 0 or left_project_center_item[0] >= img_size[0]//4: continue
                if left_project_center_item[1] < 0 or left_project_center_item[1] >= img_size[1]//4: continue

                if right_project_center_item[0] < 0 or right_project_center_item[0] >= img_size[0]//4: continue
                if right_project_center_item[1] < 0 or right_project_center_item[1] >= img_size[1]//4: continue

                # generate the radius of gaussian heatmap
                w, h = left_box[2] - left_box[0], left_box[3] - left_box[1]
                width_r = right_box[2] - right_box[0]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))

                if gt_names[i] in ['Van', 'Truck']:
                    draw_umich_gaussian(heatmap[0], left_project_center_item, radius)
                    continue

                draw_umich_gaussian(heatmap[gt_classes[i]], left_project_center_item, radius)
                # encoding 2d/3d offset & 2d size
                indices[i] = left_project_center_item[1] * self.resolution[1]//4 + left_project_center_item[0]
                offset_2d_left[i] = center_2d_left - left_project_center_item
                offset_2d_right[i] = center_2d_right - left_project_center_item
                size_2d_left[i] = 1. * w, 1. * h
                width_right[i] = 1. * width_r

                mask_2d[i] = 1

        # visual for debug
        # img_l = np.ascontiguousarray(img_l, dtype=np.uint8)
        # img_r = np.ascontiguousarray(img_r, dtype=np.uint8)
        # img_union = 0.5 * img_l + 0.5 * img_r
        # img_union = np.ascontiguousarray(img_union, dtype=np.uint8)
        # for box_item in union_bbox:
        #     # print(item)
        #     cv2.rectangle(img_union, (int(box_item[0]), int(box_item[1])), (int(box_item[2]), int(box_item[3])),
        #                   (0, 255, 0), 2)
        # cv2.imwrite("/data1/czy/3D/czy_code/LIGA_czy/data/KITTI/visual/union_{}.jpg".format(item),
        #             img_union[:, :, ::-1])
        #
        # img_l = np.ascontiguousarray(img_l, dtype=np.uint8)
        # for box_item in left_project_center:
        #     #print(item)
        #     cv2.circle(img_l, (int(box_item[0]),int(box_item[1])), 1, (0, 255, 0), 2)
        #     #cv2.rectangle(img_l, (int(box_item[0]), int(box_item[1])), (int(box_item[2]), int(box_item[3])), (0, 255, 0), 2)
        # cv2.imwrite("/data1/czy/3D/czy_code/LIGA_czy/data/KITTI/visual/left_{}.jpg".format(item), img_l[:,:,::-1])
        #
        # img_r = np.ascontiguousarray(img_r, dtype=np.uint8)
        # for box_item in right_project_center:
        #     #print(item)
        #     cv2.circle(img_r, (int(box_item[0]),int(box_item[1])), 1, (0, 255, 0), 2)
        #     #cv2.rectangle(img_r, (int(box_item[0]), int(box_item[1])), (int(box_item[2]), int(box_item[3])), (0, 255, 0), 2)
        # cv2.imwrite("/data1/czy/3D/czy_code/LIGA_czy/data/KITTI/visual/right_{}.jpg".format(item), img_r[:,:,::-1])

        #cv2.imwrite("/data1/czy/3D/czy_code/LIGA_czy/data/KITTI/visual/heatmap.jpg".format(item), (heatmap.transpose(1, 2, 0)[:,:,0]*255))

        img_l = np.array(img_l, dtype=np.float32)
        img_r = np.array(img_r, dtype=np.float32)

        if self.istrain == True:
            input['frame_id'] = index
            input['bbox_downsample_ratio'] = img_size/features_size[::-1]
            input['image_shape'] = img_l.shape[:2]
            input['calib'] = calib
            # input['left_bbox'] = left_bbox
            # input['right_bbox'] = right_bbox
            # input['union_bbox'] = union_bbox
            input['left_img'] = img_l
            input['right_img'] = img_r

            ### target ##
            input['size_2d_left'] = size_2d_left
            input['heatmap'] = heatmap
            input['offset_2d_left'] = offset_2d_left
            input['offset_2d_right'] = offset_2d_right
            input['indices'] = indices
            input['width_right']= width_right
            input['mask_2d'] = mask_2d


        else:
            input['frame_id'] = index
            input['bbox_downsample_ratio'] = img_size/features_size[::-1]
            input['image_shape'] = img_size
            input['calib'] = calib
            input['left_img'] = img_l
            input['right_img'] = img_r


        return input


    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['left_bbox' , 'right_bbox', 'union_bbox']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ['left_img', 'right_img', 'depth_gt_img', 'depth_fgmask_img']:
                    # Get largest image size (H, W)
                    if key in ['depth_gt_img', 'depth_fgmask_img']:
                        val = [np.expand_dims(x, -1) for x in val]

                    # max_h = np.max([x.shape[0] for x in val])
                    # max_w = np.max([x.shape[1] for x in val])
                    max_h = 384
                    max_w = 1248

                    padded_imgs = []
                    for i, img in enumerate(val):
                        pad_h = get_pad_params(desired_size=max_h, cur_size=img.shape[0])
                        pad_w = get_pad_params(desired_size=max_w, cur_size=img.shape[1])
                        pad_width = (pad_h, pad_w, (0, 0))

                        if key in ['left_img', 'right_img']:
                            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                            img = (img.astype(np.float32) / 255 - mean) / std

                        img = np.pad(img, pad_width, mode='constant')

                        padded_imgs.append(img)

                    ret[key] = np.stack(
                        padded_imgs, axis=0).transpose(0, 3, 1, 2)
                elif key in ['calib']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    cfg = {'root_dir': '../../../data/KITTI',
           'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.8, 'shift': 0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist': ['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center': False}
    dataset = KITTI_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break

    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
