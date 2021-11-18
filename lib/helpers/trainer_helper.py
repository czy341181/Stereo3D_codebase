import os
import tqdm

import torch
import numpy as np
import torch.nn as nn
import kornia
import time
import shutil

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from torch.nn.utils import clip_grad_norm_
from progress.bar import Bar

from lib.helpers.decode_helper import extract_dets_from_stereo_outputs
from lib.helpers.decode_helper import decode_detections


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 rank):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.cuda()

        self.class_name = self.test_loader.dataset.class_name
        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            self.epoch = load_checkpoint(model=self.model,
                                         optimizer=self.optimizer,
                                         filename=cfg['resume_model'],
                                         map_location=self.device,
                                         logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1

        if cfg['sync_bn'] == True:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # DDP
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[rank % torch.cuda.device_count()], find_unused_parameters=True)

        #self.model = torch.nn.DataParallel(self.model).cuda()

    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            #self.train_one_epoch()
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step(self.epoch)


            #save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs('checkpoints', exist_ok=True)
                ckpt_name = os.path.join('checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)

            if (self.epoch % self.cfg['eval_frequency']) == 0:
                self.inference()
            progress_bar.update()

        return None


    def train_one_epoch(self):
        self.model.train()
        loss_stats = ['seg_loss', 'offset2d_left_loss', 'offset2d_right_loss', 'size_2d_left_loss', 'width_right_loss']
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in loss_stats}
        num_iters = len(self.train_loader)
        bar = Bar('{}/{}'.format("3D", "Stereo"), max=num_iters)
        end = time.time()

        #progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, inputs in enumerate(self.train_loader):
            for key, val in inputs.items():
                if not isinstance(val, np.ndarray):
                    continue
                if key in ['frame_id', 'metadata', 'calib']:
                    continue
                # TODO
                if key in ['left_img' , 'right_img']:
                    inputs[key] = torch.from_numpy(val).float().cuda()
                    #inputs[key] = kornia.image_to_tensor(val).float().cuda()
                elif key in ['image_shape']:
                    inputs[key] = torch.from_numpy(val).int().cuda()
                else:
                    inputs[key] = torch.from_numpy(val).float().cuda()

            # train one batch
            self.optimizer.zero_grad()
            ret_dict, tb_dict  = self.model(inputs)

            loss = ret_dict.mean()
            loss.backward()
            #clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                self.epoch, batch_idx, num_iters, phase="train",
                total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    tb_dict[l], inputs['left_img'].shape[0])
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            bar.next()
        bar.finish()

    def inference(self):
        # torch.set_grad_enabled(False)
        self.model.eval()
        left_results = {}
        right_results = {}
        dataset = self.test_loader.dataset

        output_path = self.cfg['output_path']

        if os.path.exists(output_path):
            shutil.rmtree(output_path, True)
        os.makedirs(output_path, exist_ok=False)
        with torch.no_grad():
            progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
            for batch_idx, inputs in enumerate(self.test_loader):
                # load evaluation data and move data to GPU.
                for key, val in inputs.items():
                    if not isinstance(val, np.ndarray):
                        continue
                    if key in ['frame_id', 'metadata', 'calib']:
                        continue
                    # TODO
                    if key in ['left_img', 'right_img']:
                        inputs[key] = torch.from_numpy(val).float().cuda()
                        # inputs[key] = kornia.image_to_tensor(val).float().cuda()
                    elif key in ['image_shape']:
                        inputs[key] = torch.from_numpy(val).int().cuda()
                    else:
                        inputs[key] = torch.from_numpy(val).float().cuda()

                pred_dicts= self.model(inputs, False)

                dets_l, dets_r = self.process_dets2result(pred_dicts, inputs['frame_id'], inputs['bbox_downsample_ratio'])
                left_results.update(dets_l)
                right_results.update(dets_r)


                progress_bar.update()

        progress_bar.close()


        self.save_results(left_results, './left_outputs')
        self.save_results(right_results, './right_outputs')
        self.test_loader.dataset.eval(results_dir='./left_outputs/data', logger=self.logger, label_flag='left')
        self.test_loader.dataset.eval(results_dir='./right_outputs/data', logger=self.logger, label_flag='right')

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir,True)
        os.makedirs(output_dir, exist_ok=False)

        for img_id in results.keys():
            output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write(' 1.50 1.69 4.33 3.45 2.41 36.08 -1.55 2.45')  #TODO remove it
                f.write('\n')
            f.close()

    def process_dets2result(self, outputs, frame_id, bbox_downsample_ratio):
        dets_l, dets_r = extract_dets_from_stereo_outputs(outputs=outputs, K=50)
        dets_l = dets_l.detach().cpu().numpy()
        dets_r = dets_r.detach().cpu().numpy()
        # get corresponding calibs & transform tensor to numpy
        calibs = [self.test_loader.dataset.get_calib(index) for index in frame_id]
        #info = {key: val.detach().cpu().numpy() for key, val in info.items()}
        #cls_mean_size = self.test_loader.dataset.cls_mean_size
        dets_l = decode_detections(dets=dets_l,
                                 frame_id=frame_id,
                                 bbox_downsample_ratio=bbox_downsample_ratio,
                                 calibs=calibs,
                                 threshold=self.cfg.get('threshold', 0.2))
        dets_r = decode_detections(dets=dets_r,
                                 frame_id=frame_id,
                                bbox_downsample_ratio=bbox_downsample_ratio,
                                 calibs=calibs,
                                 threshold=self.cfg.get('threshold', 0.2))
        return dets_l, dets_r
