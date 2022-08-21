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


class OnePlusRggb2ObjDataset(data.Dataset):
    '''
    Read DNG image (png format) and object labels.
    '''

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.lmdb_env = None  # environment for lmdb
        self.max_crop = self.opt['max_crop']  # in case of empty targets

        # DNG and json files
        dataroot = self.opt['dataroot']
        dng_files = glob.glob(osp.join(dataroot, '*.png'))
        json_files = glob.glob(osp.join(dataroot, '*.json'))
        dng_files.sort()
        json_files.sort()
        self.dng_files = dng_files
        self.json_files = json_files

        # object dict
        # self.object_dict = {'person': 0, 'car': 1, 'bus': 2, 'truck': 3, 'traffic_light': 4}
        self.object_dict = {'person': 0, 'car': 2, 'bus': 5}  # refer to coco names
        self.object_classes = self.object_dict.keys()

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
        data_size = self.opt['data_size']

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

        # data resize to 1024x1024 (to save memory)
        # require zero padding
        # assume width > height
        desired_size = 1024
        origin_height = img_noi.shape[0]
        origin_width = img_noi.shape[1]
        image_width = desired_size  # resize resolution
        image_height = origin_height * image_width // origin_width
        image_height = image_height - image_height % 4  # must be divided by 4
        delta_h = desired_size - image_height
        top, bottom = delta_h // 2, delta_h // 2  # pad top and bottom

        ratio_height = 1. * image_height / origin_height
        ratio_width = 1. * image_width / origin_width
        img_noi_r  = img_noi[0::2, 0::2]
        img_noi_g1 = img_noi[0::2, 1::2]
        img_noi_g2 = img_noi[1::2, 0::2]
        img_noi_b  = img_noi[1::2, 1::2]
        # nearest neighbor interpolation
        ch_hei = image_height // 2
        ch_wid = image_width // 2
        img_resize_r = cv2.resize(img_noi_r, (ch_wid, ch_hei), interpolation=cv2.INTER_NEAREST)
        img_resize_g1 = cv2.resize(img_noi_g1, (ch_wid, ch_hei), interpolation=cv2.INTER_NEAREST)
        img_resize_g2 = cv2.resize(img_noi_g2, (ch_wid, ch_hei), interpolation=cv2.INTER_NEAREST)
        img_resize_b = cv2.resize(img_noi_b, (ch_wid, ch_hei), interpolation=cv2.INTER_NEAREST)
        # make border (zero padding)
        pad_top = top // 2
        pad_bottom = bottom // 2
        color = [0, 0, 0]
        img_resize_r = cv2.copyMakeBorder(img_resize_r, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)
        img_resize_g1 = cv2.copyMakeBorder(img_resize_g1, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)
        img_resize_g2 = cv2.copyMakeBorder(img_resize_g2, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)
        img_resize_b = cv2.copyMakeBorder(img_resize_b, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)
        # put channels back to image
        img_resize = np.zeros((desired_size, desired_size))
        img_resize[0::2, 0::2] = img_resize_r
        img_resize[0::2, 1::2] = img_resize_g1
        img_resize[1::2, 0::2] = img_resize_g2
        img_resize[1::2, 1::2] = img_resize_b

        # random crop
        img_noi = np.expand_dims(img_resize, axis=2)  # HW -> HWC, update img_noi
        img_noi_ori = img_noi.copy()
        # resample given empty targets
        for _ in range(self.max_crop):
            # guarantee RGGB pattern (// 2 * 2)
            rand_r = random.randint(min(top, desired_size - data_size), desired_size - data_size)  # do not crop top
            rand_c = random.randint(0, desired_size - data_size)
            rand_r = (rand_r // 2) * 2
            rand_c = (rand_c // 2) * 2
            img_noi = img_noi_ori[rand_r: rand_r+data_size, rand_c: rand_c+data_size, :]

            # process label_gt (filter valid objects)
            targets = []
            for idx in range(len(label_gt['shapes'])):
                # filter out invalid classes
                if label_gt['shapes'][idx]['label'] not in self.object_classes:
                    continue
                # coordinates
                coordinates = np.array(label_gt['shapes'][idx]['points'])
                x_min, y_min = np.min(coordinates[:, 0]), np.min(coordinates[:, 1])
                x_max, y_max = np.max(coordinates[:, 0]), np.max(coordinates[:, 1])
                x_min = x_min * ratio_width - rand_c  # x, col
                x_max = x_max * ratio_width - rand_c
                y_min = y_min * ratio_height + top - rand_r  # y, row, add top
                y_max = y_max * ratio_height + top - rand_r
                # calculate valid bbox
                x_min_valid = np.minimum(np.maximum(x_min, 0), data_size-1)
                x_max_valid = np.minimum(np.maximum(x_max, 0), data_size-1)
                y_min_valid = np.minimum(np.maximum(y_min, 0), data_size-1)
                y_max_valid = np.minimum(np.maximum(y_max, 0), data_size-1)
                # check "IoU" (keep if it is larger than 0.5)
                area_ori = (x_max - x_min) * (y_max - y_min)
                area_valid = (x_max_valid - x_min_valid) * (y_max_valid - y_min_valid)
                iou = area_valid / area_ori
                if iou >= 0.5:
                    # center and size (range [0,1))
                    x_c = (x_min_valid + x_max_valid) / 2. / data_size
                    y_c = (y_min_valid + y_max_valid) / 2. / data_size
                    delta_x = (x_max_valid - x_min_valid) / data_size
                    delta_y = (y_max_valid - y_min_valid) / data_size
                    # object class
                    object_class = self.object_dict[label_gt['shapes'][idx]['label']]
                    targets.append([0, object_class, x_c, y_c, delta_x, delta_y])
            if len(targets) > 0:
                break
        targets = np.asarray(targets, dtype=np.float32)

        # HWC -> CHW
        img_noi = np.transpose(img_noi, [2, 0, 1])

        # set data type as float32 in range [0., 1.]
        img_noi = img_noi.astype(np.float32) / 1023.

        # todo: data augmentation

        return {'noisy': img_noi, 'label': targets}

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


if __name__ == '__main__':
    dataset_opt = {
        'dataroot': '/DATA/OnePlus/1230_dng_json_first100/',
        'data_size': 1000,
        'phase': 'train',
        'mode': 'OnePlus_Rggb2Obj',
        'data_type': 'img',
        'n_workers': 0,
        'batch_size': 4,
        'max_crop': 2
    }
    opt = {
        'dist': False,
        'gpu_ids': [0]
    }
    from data import create_dataset
    train_set = create_dataset(dataset_opt)

    # train loader
    from data import create_dataloader
    train_loader = create_dataloader(train_set, dataset_opt, opt, collate_fn=train_set.collate_fn)
    for train_data in train_loader:
        imgs_noi = train_data['noisy']
        targets = train_data['label']
        img_noi = imgs_noi.numpy()[0] / 1023.
        print(imgs_noi.shape)
        print(targets)
        plt.imshow(img_noi[0] * 3)
        plt.show()
        exit(0)
