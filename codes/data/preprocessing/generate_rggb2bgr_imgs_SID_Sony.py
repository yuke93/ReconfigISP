import glob
import os
import os.path as osp
import rawpy
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

# parameters
size = 256
stride = 256

# directory
sony_dir = '/DATA/SID/Sony/'
input_dir = osp.join(sony_dir, 'short')
gt_dir = osp.join(sony_dir, 'long')
save_dir = '/DATA/SID/imgs/SID_Sony_Rggb2Bgr_size{}/'.format(size)
if not osp.exists(save_dir):
    os.mkdir(save_dir)

# get train IDs
train_fns = glob.glob(osp.join(gt_dir, '0*.ARW'))  # bug fixed, should use gt_dir rather than input dir
print(len(train_fns))
train_ids = [int(osp.basename(train_fn)[:5]) for train_fn in train_fns]
train_ids.sort()

# generate data
keys_noisy = []
keys_ratio = []
keys_gt = []
for id in train_ids:
    in_files = glob.glob(osp.join(input_dir, '{:05d}'.format(id) + '_*.ARW'))
    gt_path = glob.glob(osp.join(gt_dir, '{:05d}'.format(id) + '_*.ARW'))[0]  # the same GT for different inputs
    print('processing gt image {}'.format(gt_path))
    for k, in_path in enumerate(in_files):
        # file name
        in_fn = osp.basename(in_path)
        gt_fn = osp.basename(gt_path)

        # exposure time
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = gt_exposure / in_exposure

        # read image
        in_raw = rawpy.imread(in_path)
        in_raw_preprocess = in_raw.raw_image_visible.astype(np.float32)
        # subtract the black level (no need when processed with rawpy to get RGB)
        in_raw_preprocess = np.maximum(in_raw_preprocess - 512, 0) / (16383 - 512)

        # input raw images: noisy, ratio
        raw_noisy = np.minimum(np.maximum(in_raw_preprocess, 0.), 1.)
        raw_noisy = (raw_noisy * 16383).astype(np.uint16)
        raw_ratio = np.minimum(np.maximum(in_raw_preprocess * ratio, 0.), 1.)
        raw_ratio = (raw_ratio * 16383).astype(np.uint16)

        # ground-truth RGB image: gt (only save once)
        if k == 0:
            gt_raw = rawpy.imread(gt_path)
            gt_rgb = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_rgb = np.float32(gt_rgb / 65535.0)

            # save BGR image with uint8
            gt_bgr = gt_rgb[:, :, ::-1]
            gt_bgr = (gt_bgr * 255).astype(np.uint8)

        # image size
        row, col = raw_noisy.shape
        assert gt_bgr.shape[0] == row
        assert gt_bgr.shape[1] == col

        # save patches
        count_patch = 0
        for r in range(0, row - size, stride):
            for c in range(0, col - size, stride):
                count_patch += 1

                # noisy raw
                patch_noisy = raw_noisy[r: r+size, c: c+size]
                key_noisy = osp.splitext(in_fn)[0] + '_{:04d}_noisy.png'.format(count_patch)
                keys_noisy.append(key_noisy)
                name_noisy = osp.join(save_dir, key_noisy)
                cv2.imwrite(name_noisy, patch_noisy)

                # ratio raw
                patch_ratio = raw_ratio[r: r+size, c: c+size]
                key_ratio = osp.splitext(in_fn)[0] + '_{:04d}_ratio.png'.format(count_patch)
                keys_ratio.append(key_ratio)
                name_ratio = osp.join(save_dir, key_ratio)
                cv2.imwrite(name_ratio, patch_ratio)

                # gt BGR (only save once)
                key_gt = osp.splitext(gt_fn)[0] + '_{:04d}_gt.png'.format(count_patch)
                keys_gt.append(key_gt)
                if k == 0:
                    patch_gt = gt_bgr[r: r+size, c: c+size, :]
                    name_gt = osp.join(save_dir, key_gt)
                    cv2.imwrite(name_gt, patch_gt)

#### create meta information
meta_info = dict()
meta_info['resolution'] = size
meta_info['keys_noisy'] = keys_noisy
meta_info['keys_ratio'] = keys_ratio
meta_info['keys_gt'] = keys_gt
print('Saving meta data ...')
# pickle dump
pickle.dump(meta_info, open(osp.join(save_dir, 'meta_info.pkl'), 'wb'))
print('Done!')
