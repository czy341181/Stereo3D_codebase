import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger


parser = argparse.ArgumentParser(description='End-to-End Stereo 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
args = parser.parse_args()



def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))
    log_file = 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = create_logger(log_file)


    # build dataloader
    train_loader, test_loader  = build_dataloader(cfg['dataset'])

    # build model
    model = build_model(cfg['model'])

    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger)
        tester.test()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr scheduler
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=cfg['trainer']['max_epoch'],
        last_epoch=-1, optim_cfg=cfg['optimizer']
    )

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d'  % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f'  % (cfg['optimizer']['lr']))
    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=lr_warmup_scheduler,
                      logger=logger)
    trainer.train()

    # logger.info('###################  Evaluation  ##################' )
    # tester = Tester(cfg=cfg['tester'],
    #                 model=model,
    #                 dataloader=test_loader,
    #                 logger=logger)
    # tester.test()


if __name__ == '__main__':
    main()