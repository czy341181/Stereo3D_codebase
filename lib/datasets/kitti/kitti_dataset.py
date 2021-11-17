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

import cv2

#from lib.models.roiaware_pool3d import roiaware_pool3d_utils

import scipy.misc


class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):
        # basic configuration
        self.root_dir = cfg.get('root_dir', '../../data/KITTI')
        self.split = split
        self.num_classes = len(cfg['writelist'])
        self.class_name = cfg['writelist']
        self.cls2id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'object', 'testing' if split == 'test' else 'training')
        # self.image_dir = os.path.join(self.data_dir, 'image_2')
        # self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # data augmentation configuration
        self.istrain = True if split in ['train', 'trainval'] else False

        # monocular
        self.downsample = 4
        self.depth_downsample_factor = 4
        self.point_cloud_range = cfg['pc_range']

        # stereo
        self.use_van = True
        self.use_person_sitting = True
        self.flip_type = cfg['flip_type']
        self.crop_size = cfg['crop_size']
        self.gen_depth = cfg['gen_depth']

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


    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

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

        calib = self.get_calib(index)
        calib_ori = copy.deepcopy(calib)
        if self.istrain == True:
            l_objects = self.get_left_label(index)
            r_objects = self.get_right_label(index)

            gt_names = []
            gt_truncated = []
            gt_occluded = []

            locations = []
            dims = []
            rotations_y = []
            left_bbox = []
            right_bbox = []
            for l_object, r_object in zip(l_objects, r_objects):
                if self.use_van:  # default: True
                    # Car 14357, Van 1297
                    if l_object.cls_type == 'Van':
                        l_object.cls_type = 'Car'

                if self.use_person_sitting:  # default: True
                    # Ped 2207, Person_sitting 56
                    if l_object.cls_type == 'Person_sitting':
                        l_object.cls_type = 'Pedestrian'
                if l_object.cls_type not in self.class_name:
                    continue

                if float(l_object.trucation) >= 0.98:
                    continue

                # union_bbox_item = np.zeros((4))
                # union_bbox_item[0] = min(l_object.box2d[0], r_object.box2d[0])
                # union_bbox_item[1] = min(l_object.box2d[1], r_object.box2d[1])
                # union_bbox_item[2] = max(l_object.box2d[2], r_object.box2d[2])
                # union_bbox_item[3] = max(l_object.box2d[3], r_object.box2d[3])


                gt_names.append(l_object.cls_type)
                locations.append(l_object.pos)
                dims.append([l_object.l, l_object.h, l_object.w])
                rotations_y.append(l_object.ry)
                left_bbox.append(l_object.box2d)
                right_bbox.append(r_object.box2d)
                gt_truncated.append(l_object.trucation)
                gt_occluded.append(l_object.occlusion)

            gt_names = np.array(gt_names)
            locations = np.array(locations)
            dims = np.array(dims)
            rotations_y = np.array(rotations_y)
            left_bbox = np.array(left_bbox)
            right_bbox = np.array(right_bbox)
            gt_truncated = np.array(gt_truncated)
            gt_occluded = np.array(gt_occluded)
            gt_classes = np.array([self.cls2id[n] + 1 for n in gt_names], dtype=np.int32)

        if self.istrain == True:
            img_l, img_r, left_bbox, right_bbox, calib, flip_this_image = \
                random_flip(img_l, img_r, left_bbox, right_bbox, calib)

            union_bbox = copy.copy(left_bbox)
            for i in range(union_bbox.shape[0]):
                union_bbox[i, 0] = min(left_bbox[i, 0], right_bbox[i, 0])
                union_bbox[i, 1] = min(left_bbox[i, 1], right_bbox[i, 1])
                union_bbox[i, 2] = max(left_bbox[i, 2], right_bbox[i, 2])
                union_bbox[i, 3] = max(left_bbox[i, 3], right_bbox[i, 3])


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
        # for box_item in left_bbox:
        #     #print(item)
        #     cv2.rectangle(img_l, (int(box_item[0]), int(box_item[1])), (int(box_item[2]), int(box_item[3])), (0, 255, 0), 2)
        # cv2.imwrite("/data1/czy/3D/czy_code/LIGA_czy/data/KITTI/visual/left{}.jpg".format(item), img_l[:,:,::-1])
        #
        # img_r = np.ascontiguousarray(img_r, dtype=np.uint8)
        # for box_item in right_bbox:
        #     #print(item)
        #     cv2.rectangle(img_r, (int(box_item[0]), int(box_item[1])), (int(box_item[2]), int(box_item[3])), (0, 255, 0), 2)
        # cv2.imwrite("/data1/czy/3D/czy_code/LIGA_czy/data/KITTI/visual/right_{}.jpg".format(item), img_r[:,:,::-1])
        img_l = np.array(img_l, dtype=np.float32)
        img_r = np.array(img_r, dtype=np.float32)

        if self.istrain == True:
            input['frame_id'] = index
            input['image_shape'] = img_l.shape[:2]
            input['calib'] = calib
            input['left_bbox'] = left_bbox
            input['right_bbox'] = right_bbox
            input['union_bbox'] = union_bbox
            input['left_img'] = img_l
            input['right_img'] = img_r


        else:
            input['frame_id'] = index
            input['image_shape'] = img_size
            input['calib'] = calib
            input['left_img'] = img_l
            input['right_img'] = img_r


        return input

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()

            # NOTE: in stereo mode, the 3d boxes are predicted in pseudo lidar coordinates
            pred_boxes_camera = boxes3d_pselidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape, fix_neg_z_bug=True
            )

            ###  monocular lidar coordinate ###
            # pred_boxes_camera = boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            # pred_boxes_img = boxes3d_kitti_camera_to_imageboxes(
            #     pred_boxes_camera, calib, image_shape=image_shape
            # )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            # pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path + '/' + ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

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
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
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
