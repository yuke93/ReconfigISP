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
import glob
import os.path as osp
import json
try:
    sys.path.append('/mnt/lustre/share/pymc/py3/')
    import mc
except:
    pass


class OnePlusRggb2ObjTestDataset(data.Dataset):
    '''
    Read DNG image (png format) and object labels.
    '''

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.lmdb_env = None  # environment for lmdb

        # DNG and json files
        dataroot = self.opt['dataroot']
        dng_files = glob.glob(osp.join(dataroot, '*.png'))
        json_files = glob.glob(osp.join(dataroot, '*.json'))
        dng_files.sort()
        json_files.sort()
        self.dng_files = dng_files
        self.json_files = json_files

        # object dict
        self.object_dict = {'person': 0, 'car': 1, 'bus': 2, 'truck': 3, 'traffic_light': 4}

        # mclient
        if self.data_type == 'mc':
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Invalid data type {}'.format(self.data_type))
            
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

    def _read_json_mc(self, path):
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_str = mc.ConvertString(value)
        value_json = json.loads(value_str)
        return value_json

    def __getitem__(self, index):
        if self.data_type == 'mc':
            if self.mclient is None:
                self._init_memcached()
        elif self.data_type == 'img':
            pass

        # get image and label
        dng_file = self.dng_files[index]
        json_file = self.json_files[index]
        dataroot = self.opt['dataroot']
        if self.data_type == 'mc':
            # read noisy image
            img_noi = self._read_img_mc(dng_file, np.uint16)  # uint16
            # read gt label
            label_gt = self._read_json_mc(json_file)
        elif self.data_type == 'img':
            # read noisy image
            img_noi = cv2.imread(dng_file, cv2.IMREAD_UNCHANGED)
            # read gt label
            with open(json_file) as fj:
                label_gt = json.load(fj) 
        img_noi = np.expand_dims(img_noi, axis=2)  # HW -> HWC

        # random crop
        image_height = img_noi.shape[0]
        image_width = img_noi.shape[1]

        # process label_gt (filter valid objects)
        targets = []
        for idx in range(len(label_gt['shapes'])):
            # coordinates
            coordinates = np.array(label_gt['shapes'][idx]['points'])
            x_min, y_min = np.min(coordinates[:, 0]), np.min(coordinates[:, 1])
            x_max, y_max = np.max(coordinates[:, 0]), np.max(coordinates[:, 1])
            # center and size (range [0,1))
            x_c = (x_min + x_max) / 2. / image_width
            y_c = (y_min + y_max) / 2. / image_height
            delta_x = (x_max - x_min) / image_width
            delta_y = (y_max - y_min) / image_height
            # object class
            object_class = self.object_dict[label_gt['shapes'][idx]['label']]
            targets.append([0, object_class, x_c, y_c, delta_x, delta_y])
        targets = np.asarray(targets, dtype=np.float32)

        # HWC -> CHW
        img_noi = np.transpose(img_noi, [2, 0, 1])

        # set data type as float32 in range [0., 1.]
        img_noi = img_noi.astype(np.float32) / 1023.

        # input image name
        name, _ = osp.splitext(osp.basename(dng_file))

        return {'noisy': img_noi, 'label': targets, 'name': name}

    def collate_fn(self, batch):
        # get data
        img_noi = [sample['noisy'] for sample in batch]
        targets_ori = [sample['label'] for sample in batch]

        # noisy image
        img_noi = torch.Tensor(img_noi)

        # label
        targets = []
        for idx, boxes_ori in enumerate(targets_ori):
            # add sample index to targets
            # todo: need to check, original implementation filters empty placeholders prior to assigning index
            if len(boxes_ori) > 0:
                boxes = torch.Tensor(boxes_ori)
                boxes[:, 0] = idx
                targets.append(boxes)
        if len(targets) > 0:
            targets = torch.cat(targets, 0)  # cat instead of stack
        else:
            targets = torch.Tensor(targets)  # empty tensor        

        return {'noisy': torch.Tensor(img_noi), 'label': targets}


    def __len__(self):
        return len(self.dng_files)

