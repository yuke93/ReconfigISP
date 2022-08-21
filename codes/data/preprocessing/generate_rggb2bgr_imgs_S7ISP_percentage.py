import glob
import os
import os.path as osp
import rawpy
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import pdb

# parameters
size = 256
stride = 256
percentage = 5
index_folder = '/DATA/S7-ISP-Index/'
data_folder = '/DATA/S7-ISP-Dataset/'
save_folder = '/DATA/S7-ISP-Rggb2Bgr_{}pct_size{}/'.format(percentage, size)

# make save folder
if not osp.exists(save_folder):
    os.mkdir(save_folder)

# read image list
train_list = osp.join(index_folder, 'train_' + str(percentage) + 'percent.txt')
with open(train_list) as f:
    train_files = f.readlines()
train_files = [file.split('\n')[0] for file in train_files]

# all images list
all_files = glob.glob(osp.join(data_folder, '*'))
all_files.sort()

# for loop
count = 0
keys_noisy = []
keys_gt = []
for cur_file in all_files:
    cur_name = cur_file.split('/')[-1]
    if cur_name in train_files:
        count += 1
        print('Process {} (No. {})'.format(cur_name, count))
    else:
        continue

    # image paths
    in_path = osp.join(data_folder, cur_name, 'short_exposure1.dng')
    gt_path = osp.join(data_folder, cur_name, 'medium_exposure.jpg')

    # read input image
    in_raw = rawpy.imread(in_path)
    # check GRBG pattern
    assert in_raw.color_desc == b'RGBG'
    assert in_raw.raw_pattern[0][1] == 0
    in_raw_preprocess = in_raw.raw_image_visible.astype(np.uint16)
    # check [0, 1023]
    assert in_raw_preprocess.max() <= 1023
    # read gt
    gt_bgr = cv2.imread(gt_path)

    # crop to RGGB
    raw_noisy = in_raw_preprocess[:, 1:]
    gt_bgr = gt_bgr[:, 1:, :]

    # image size
    row, col = raw_noisy.shape
    try:
        assert gt_bgr.shape[0] == row
        assert gt_bgr.shape[1] == col
    except:
        # plot two images
        plt.subplot(1, 2, 1)
        plt.imshow(raw_noisy / 1023.)
        plt.subplot(1, 2, 2)
        plt.imshow(gt_bgr / 255.)
        plt.show()
        pdb.set_trace()

    # save patches
    count_patch = 0
    for r in range(0, row - size, stride):
        for c in range(0, col - size, stride):
            count_patch += 1

            # noisy raw
            patch_noisy = raw_noisy[r: r + size, c: c + size]
            key_noisy = cur_name + '_{:04d}_noisy.png'.format(count_patch)
            keys_noisy.append(key_noisy)
            name_noisy = osp.join(save_folder, key_noisy)
            cv2.imwrite(name_noisy, patch_noisy)

            # gt BGR
            key_gt = cur_name + '_{:04d}_gt.png'.format(count_patch)
            keys_gt.append(key_gt)
            patch_gt = gt_bgr[r: r + size, c: c + size, :]
            name_gt = osp.join(save_folder, key_gt)
            cv2.imwrite(name_gt, patch_gt)

#### create meta information
meta_info = dict()
meta_info['resolution'] = size
meta_info['keys_noisy'] = keys_noisy
meta_info['keys_gt'] = keys_gt
print('Saving meta data ...')
# pickle dump
pickle.dump(meta_info, open(osp.join(save_folder, 'meta_info.pkl'), 'wb'))
print('Done!')
