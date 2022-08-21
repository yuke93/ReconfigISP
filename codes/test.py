import os
import os.path as osp
import math
import argparse
import random
import logging
import torch
import cv2
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
import pdb
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
from utils.util import psnr


#### options
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, help='Path to option YMAL file.')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['mode'], len(test_set)))
    test_loaders.append(test_loader)

#### create model
model = create_model(opt)

#### random seed
seed = opt['test_seed']
if seed is None:
    seed = random.randint(1, 10000)
util.set_random_seed(seed)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['mode']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    all_psnrs_in = []
    all_psnrs_out = []

    for idx, data in enumerate(test_loader):
        print ('Image No. {}'.format(idx + 1))

        # input data and params
        img_in = data['noisy']
        img_gt = data['gt']

        # inference
        model.feed_data((img_in, img_gt))
        img_out, mid_results = model.test()

        # middle results
        middle_results = [util.tensor2bgr(img) for img in mid_results]

        # PSNR
        img_in_bgr = util.tensor2bgr(img_in)  # uint8, HxWxC
        # Bayer to BGR
        if img_in_bgr.shape[2] == 1:
            # gray image, for visualization
            img_in_bgr = np.concatenate([img_in_bgr] * 3, axis=2)  # HxWx3

        img_out_bgr = util.tensor2bgr(img_out)
        img_gt_bgr = util.tensor2bgr(img_gt)
        cur_psnr_in = psnr(img_in_bgr, img_gt_bgr)
        cur_psnr_out = psnr(img_out_bgr, img_gt_bgr)
        all_psnrs_in.append(cur_psnr_in)
        all_psnrs_out.append(cur_psnr_out)

        # in case middle results are in bayer
        for k in range(len(middle_results)):
            cur_results = middle_results[k]
            if cur_results.shape[2] == 1:
                middle_results[k] = np.concatenate([cur_results] * 3, axis=2)

        vis_imgs = [img_in_bgr] + middle_results + [img_gt_bgr]
        img_vis = np.concatenate(vis_imgs, axis=1)
        save_path = osp.join(dataset_dir, '{:03d}'.format(idx+1) + '.png')
        cv2.imwrite(save_path, img_vis)

    # print results
    all_psnrs_in = np.asarray(all_psnrs_in)
    all_psnrs_out = np.asarray(all_psnrs_out)
    print('PSNR in: min {}, max {}, mean {}, std {}'.format(
        all_psnrs_in.min(), all_psnrs_in.max(), all_psnrs_in.mean(), all_psnrs_in.std()))
    print('PSNR out: min {}, max {}, mean {}, std {}'.format(
        all_psnrs_out.min(), all_psnrs_out.max(), all_psnrs_out.mean(), all_psnrs_out.std()))
