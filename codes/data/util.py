import os
import math
import pickle
import random
import numpy as np
import torch
import cv2


###################### read images ######################
def _read_img_lmdb(env, key, size, dtype):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=dtype)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, size=None, dtype=np.uint8):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    assert env is not None
    img = _read_img_lmdb(env, path, size, dtype)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


### resize RGGB image ###
def resize_rggb(img, height, width):
    """ Resize RGGB image
    :param img: a bayer image (H, W)
    :param height: resize height
    :param width: resize width
    :return: resized image
    """

    # split image into R G G B channels
    img_r = img[0::2, 0::2]
    img_g1 = img[0::2, 1::2]
    img_g2 = img[1::2, 0::2]
    img_b = img[1::2, 1::2]

    # nearest neighbor interpolation
    ch_hei = height // 2
    ch_wid = width // 2
    img_resize_r = cv2.resize(img_r, (ch_wid, ch_hei), interpolation=cv2.INTER_NEAREST)
    img_resize_g1 = cv2.resize(img_g1, (ch_wid, ch_hei), interpolation=cv2.INTER_NEAREST)
    img_resize_g2 = cv2.resize(img_g2, (ch_wid, ch_hei), interpolation=cv2.INTER_NEAREST)
    img_resize_b = cv2.resize(img_b, (ch_wid, ch_hei), interpolation=cv2.INTER_NEAREST)

    # put channels back to image
    img_resize = np.zeros((height, width))
    img_resize[0::2, 0::2] = img_resize_r
    img_resize[0::2, 1::2] = img_resize_g1
    img_resize[1::2, 0::2] = img_resize_g2
    img_resize[1::2, 1::2] = img_resize_b

    return img_resize
