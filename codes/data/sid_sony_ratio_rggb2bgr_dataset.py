import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import pdb
import sys
try:
    sys.path.append('/mnt/lustre/share/pymc/py3/')
    import mc
except:
    pass


class SidSonyRatioRggb2BgrDataset(data.Dataset):
    '''
    Read Noisy (times ratio) and GT image pairs.
    '''
    #todo: should correct brightness at last for this dataset

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.lmdb_env = None  # environment for lmdb

        # get meta info
        meta_path = osp.join(self.opt['dataroot'], 'meta_info.pkl')
        with open(meta_path, 'rb') as f:
            meta_info = pickle.load(f)
        self.keys_noisy = meta_info['keys_ratio']  # load data "ratio"
        self.keys_gt = meta_info['keys_gt']
        self.image_size = meta_info['resolution']

        # mclient
        if self.data_type == 'lmdb':
            pass
        elif self.data_type == 'mc':
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Invalid data type {}'.format(self.data_type))
            
        # select exposure time
        self.sid_expo_in = self.opt['sid_expo_in']
        self.sid_expo_gt = self.opt['sid_expo_gt']
        if self.sid_expo_in is None and self.sid_expo_gt is None:
            pass
        else:
            keys_noisy = [k_in for k_in, k_gt in zip(self.keys_noisy, self.keys_gt) if self.sid_expo_in in k_in and self.sid_expo_gt in k_gt]
            keys_gt = [k_gt for k_in, k_gt in zip(self.keys_noisy, self.keys_gt) if self.sid_expo_in in k_in and self.sid_expo_gt in k_gt]
            self.keys_noisy, self.keys_gt = keys_noisy, keys_gt

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.lmdb_env = lmdb.open(self.opt['dataroot'], readonly=True, lock=False, readahead=False, meminit=False)

    def _init_memcached(self):
        # specify the config files
        server_list_config_file = '/mnt/lustre/share/memcached_client/server_list.conf'
        client_config_file = '/mnt/lustre/share/memcached_client/client.conf'
        self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                      client_config_file)

    def _read_img_mc(self, path, dtype):
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        assert img.dtype == dtype
        return img

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if self.lmdb_env is None:
                self._init_lmdb()
        elif self.data_type == 'mc':
            if self.mclient is None:
                self._init_memcached()
        elif self.data_type == 'img':
            pass
        data_size = self.opt['data_size']

        # get images
        key_noi = self.keys_noisy[index]
        key_gt = self.keys_gt[index]
        if self.data_type == 'lmdb':
            resolution_bgr = (3, self.image_size, self.image_size)
            resolution_rggb = (1, self.image_size, self.image_size)
            img_noi = util.read_img(self.lmdb_env, key_noi, size=resolution_rggb, dtype=np.int16)  # int16
            img_gt = util.read_img(self.lmdb_env, key_gt, size=resolution_bgr, dtype=np.uint8)  # uint8
        elif self.data_type == 'mc':
            dataroot = self.opt['dataroot']
            # read noisy image
            noisy_name = osp.join(dataroot, key_noi)
            img_noi = self._read_img_mc(noisy_name, np.uint16)  # uint16
            img_noi = np.expand_dims(img_noi, axis=2)  # HW -> HWC
            # read gt image
            gt_name = osp.join(dataroot, key_gt)
            img_gt = self._read_img_mc(gt_name, np.uint8)  # uint8
        elif self.data_type == 'img':
            # read noisy image
            dataroot = self.opt['dataroot']
            noisy_name = osp.join(dataroot, key_noi)
            img_noi = cv2.imread(noisy_name, cv2.IMREAD_UNCHANGED)
            img_noi = np.expand_dims(img_noi, axis=2)  # HW -> HWC
            # read gt image
            gt_name = osp.join(dataroot, key_gt)
            img_gt = cv2.imread(gt_name, cv2.IMREAD_UNCHANGED)

        # random crop
        # guarantee RGGB pattern (// 2 * 2)
        rand_r = random.randint(0, self.image_size - data_size)
        rand_c = random.randint(0, self.image_size - data_size)
        rand_r = (rand_r // 2) * 2
        rand_c = (rand_c // 2) * 2
        img_noi = img_noi[rand_r: rand_r+data_size, rand_c: rand_c+data_size, :]
        img_gt = img_gt[rand_r: rand_r+data_size, rand_c: rand_c+data_size, :]

        # HWC -> CHW
        img_noi = np.transpose(img_noi, [2, 0, 1])
        img_gt = np.transpose(img_gt, [2, 0, 1])

        # set data type as float32 in range [0., 1.]
        img_noi = img_noi.astype(np.float32) / 16383.
        img_gt = img_gt.astype(np.float32) / 255.

        return {'noisy': img_noi, 'gt': img_gt}

    def __len__(self):
        return len(self.keys_gt)
