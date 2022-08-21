import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# Pytorch Tensor <-> state
####################
def state2tensor(state):
    """
    :param state: numpy array (NHWC), 10 bit
    :return: Pytorch Tensor (NCHW)
    """
    state = state.copy().astype(np.float32) / 1023.
    tensor = torch.Tensor(np.transpose(state, [0, 3, 1, 2]))
    return tensor


def tensor2state(tensor):
    """
    :param tensor: Pytorch Tensor (NCHW)
    :return: numpy array (NHWC), 10 bit
    """
    state = np.transpose(tensor.numpy().copy(), [0, 2, 3, 1])
    state *= 1023
    state = state.astype(np.int16)
    state = np.maximum(state, 0)  # <0 may cause error while processing
    return state


def tensor2bgr(tensor, is_uint8=True):
    """
    :param tensor: 1CHW tensor [0., 1.]
    :param is_uint8: whether to output uint8 (if not, output original dtype)
    :return: BGR image [0, 255] if is_uint8 else [0., 1.]
    """
    image = tensor.detach().cpu().numpy()
    if image.ndim == 4:
        image = image[0]
    image = np.transpose(image, axes=[1, 2, 0])  # HWC, BGR image

    if is_uint8:
        image = np.clip(image * 255, 0, 255)
        image = image.astype(np.uint8)
    else:
        pass

    return image.copy()


####################
# Metric (e.g., PSNR)
####################
def psnr(img1, img2):
    # 10 bit image
    if img1.dtype == np.int16:
        img1 = img1.astype(np.float32) / 1023.
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.
    if img2.dtype == np.int16:
        img2 = img2.astype(np.float32) / 1023.
    if img2.dtype == np.uint8:
        img2 = img2.astype(np.float32) / 255.

    mse = (img1 - img2) ** 2
    mse = mse.mean()
    return 10 * math.log10(1. / mse)
