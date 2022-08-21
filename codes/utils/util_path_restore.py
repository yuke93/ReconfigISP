import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim


def get_mse_psnr(x, y):
    if x.ndim == 4:
        mse_list = []
        psnr_list = []
        data_len = len(x)
        for k in range(data_len):
            img = x[k]
            ref = y[k]
            mse = np.mean((img - ref) ** 2)
            psnr = 10 * np.log10(1. / mse)
            mse_list.append(mse)
            psnr_list.append(psnr)
        return np.asarray(mse_list), np.asarray(psnr_list)
    elif x.ndim == 3:
        mse = np.mean((x - y) ** 2)
        psnr = 10 * np.log10(1. / mse)
        return mse, psnr
    else:
        raise ValueError('Invalid data!')


def get_ssim(x, y):
    """
    :param x: input 
    :param y: reference
    :return: SSIM
    """
    if x.ndim == 4:
        ssim_list = []
        data_len = len(x)
        for k in range(data_len):
            img = x[k]
            ref = y[k]
            ssim_value = ssim(ref, img, data_range=img.max()-img.min(), multichannel=True)
            ssim_list.append(ssim_value)
        return np.asarray(ssim_list)
    elif x.ndim == 3:
        ssim_value = ssim(y, x, data_range=x.max() - x.min(), multichannel=True)
        return ssim_value


def create_patch_mask(size, edge):
    """ create a map with all ones, except that pixels near edge approach 0 linearly
    :param size: (h, w)
    :param edge: (eh, ew)
    :return: a edge_map of size (h, w)
    """
    h, w = size
    eh, ew = edge
    assert eh <= h//2, ew <= w//2
    edge_map = np.ones((h, w), dtype=np.float32)
    for idx_h in range(eh):
        edge_map[idx_h, :] = 1. * (idx_h + 1) / (eh + 1)
        edge_map[-1 - idx_h, :] = 1. * (idx_h + 1) / (eh + 1)
    for idx_w in range(ew):
        temp_column = np.ones((h, ), dtype=np.float32) * (idx_w + 1) / (ew + 1)
        edge_map[:, idx_w] = np.minimum(edge_map[:, idx_w], temp_column)
        edge_map[:, -1 - idx_w] = np.minimum(edge_map[:, -1 - idx_w], temp_column)
    return edge_map


def whole2patch(img, size, stride, is_mask=True):
    """ split a whole image to overlapped patches
    :param img: an input color image
    :param size: (h, w), size of each patch
    :param stride: (sh, sw), stride of each patch
    :param is_mask: use edge mask or not
    :return: (patches, positions, count_map)
    """
    h, w = size
    sh, sw = stride
    H, W, C = img.shape
    assert sh<=h<=H and sw<=w<=W and C>=1
    count_map = np.zeros((H, W), dtype=np.float32)
    if is_mask:
        eh = (h - sh) // 2
        ew = (w - sw) // 2
        mask = create_patch_mask((h, w), (eh, ew))

    # crop
    patches = []
    positions = []
    h_list = list(range(0, H-h, sh)) + [H-h]
    w_list = list(range(0, W-w, sw)) + [W-w]
    for idx_h in h_list:
        for idx_w in w_list:
            # position
            positions.append([idx_h, idx_w])

            # count map
            if is_mask:
                count_map[idx_h: idx_h + h, idx_w: idx_w + w] += mask
            else:
                count_map[idx_h: idx_h + h, idx_w: idx_w + w] += 1

            # patches
            patches.append(img[idx_h: idx_h + h, idx_w: idx_w + w, :])
    positions = np.asarray(positions)
    patches = np.asarray(patches)
    return patches, positions, count_map


def patch2whole(patches, positions, count_map, stride, is_mask=True):
    """ this is the inverse function of `whole2patch`
    :param patches: cropped patches
    :param positions: position for each cropped patch
    :param count_map: how many times each pixel is counted
    :param stride: (sw, sh)
    :param is_mask: whether the count map is calculated with edge mask
    :return: image
    """
    H, W = count_map.shape  # image shape
    h, w = patches.shape[1:3]  # patch shape
    C = patches.shape[3]  # support any number of channel
    if is_mask:
        sh, sw = stride
        eh = (h - sh) // 2
        ew = (w - sw) // 2
        mask = create_patch_mask((h, w), (eh, ew))
        mask = np.repeat(np.expand_dims(mask, axis=2), C, axis=2)
    image = np.zeros((H, W, 3), dtype=np.float32)
    for patch, pos in zip(patches, positions):
        idx_h, idx_w = pos
        if is_mask:
            image[idx_h: idx_h + h, idx_w: idx_w + w, :] += patch * mask
        else:
            image[idx_h: idx_h + h, idx_w: idx_w + w, :] += patch
    image /= np.repeat(np.expand_dims(count_map, axis=2), C, axis=2)
    return image
