import os
import tqdm

import torch

from lib.helpers.save_helper import load_checkpoint




class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, eval=False):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = './outputs'
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.eval = eval


    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single':
            assert os.path.exists(self.cfg['checkpoint'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=self.cfg['checkpoint'],
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            self.evaluate()


    def inference(self):
        # torch.set_grad_enabled(False)
        self.model.eval()
        rgb_results = {}
        dataset = self.dataloader.dataset
        class_names = dataset.class_name

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

                _ = dataset.generate_prediction_dicts(
                    inputs, pred_dicts, class_names,
                    output_path=output_path
                )

                progress_bar.update()

        progress_bar.close()

        self.test_loader.dataset.eval(results_dir=output_path, logger=self.logger)

