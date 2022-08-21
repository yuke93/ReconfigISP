# tools -- original (differentiable) modules
# white balance: grayworld, manual
# gamma: gamma_manual

import sys
import os
# add system path
if os.path.exists('/DATA/ISP_Kernels/'):
    sys.path.append('/DATA/ISP_Kernels/')
import torch
import torch.nn as nn
import torch.nn.functional as F
import whitebalance as wb
import gamma as gm
import demosaic as dm
import globaltonemapping as gtm
import spatialnoisereduction as snr
import numpy as np
import pdb


class Grayworld(nn.Module):  # output is clipped to [0, 1]
    def __init__(self):
        super().__init__()
        self.kernel = wb.WhiteBalance()

    def forward(self, img, params=None):
        """
        :param img: input images (N, C, H, W)
        :param params: None
        :return: output images
        """
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)

        option = 'grayworld'
        params_dict = {
            'input': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'output': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8}
            }

        output = self.kernel.run(img, option, params_dict)

        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


class Gamma(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = gm.Gamma()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: gamma (N, 1)
        :return: output images
        """
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)

        # gamma, [0, 1]
        gamma = params

        option = 'manual'
        params_dict = {
            'gamma': gamma
            }

        output = self.kernel.run(img, option, params_dict)

        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


# conditional module base class (sRGB domain)
class ConditionalModuleBGR(nn.Module):
    def __init__(self, in_channels, out_channel):
        """
        :param in_channels: (tuple) number of input channels for each layer
        :param out_channel: (int) output channel -- number of module parameters
        """
        super().__init__()

        # check validity
        # input channel can be divided by 3
        assert in_channels[0] % 3 == 0
        self.hist_bin = in_channels[0] // 3  # bins for image channel histogram

        # activation
        self.activation_relu = nn.ReLU()
        self.activation_sigmoid = nn.Sigmoid()

        # calculate total number of parameters (exclude module parameters)
        self.total_params = 0
        self.in_out_channels = [ch for ch in in_channels] + [out_channel]
        for idx in range(len(in_channels)):
            in_ch = self.in_out_channels[idx]
            out_ch = self.in_out_channels[idx+1]
            self.total_params += in_ch * out_ch + out_ch  # linear layer (weight and bias)
        self.total_params += out_channel  # global gamma parameter

        # number of module parameters
        self.module_params = out_channel

    def forward(self, img, params):
        pass

    def _fc_forward(self, img, params):
        # check validity of parameters and image
        # print(params.size(0), self.total_params)
        assert params.size(0) == self.total_params
        assert img.size(1) == 3 # BGR 3 channels

        # batch size
        N = img.size(0)

        # histogram 
        # (N, C, H, W) --> (N, input channel)
        img_hist = []
        for cur_img in img:
            cur_hist = []
            for ch in cur_img:
                channel_hist = torch.histc(ch.detach().cpu(), bins=self.hist_bin, min=0, max=1).cuda()
                cur_hist.append(channel_hist)
            cur_hist = torch.cat(cur_hist)  # (input_channel,)
            img_hist.append(cur_hist)
        img_hist = torch.stack(img_hist)  # (N, input_channel)
        img_hist = img_hist.detach()  # no gradient for histogram

        # fully-connected layers
        par_idx = 0
        layers = len(self.in_out_channels) - 1
        hist_feat = img_hist  # histogram features
        for idx in range(layers):
            in_ch = self.in_out_channels[idx]
            out_ch = self.in_out_channels[idx+1]
            # weight
            weight_size = in_ch * out_ch
            weight = params[par_idx: par_idx + weight_size]
            weight = weight.view(in_ch, out_ch)
            par_idx += weight_size
            # bias
            bias_size = out_ch
            bias = params[par_idx: par_idx + bias_size]
            bias = bias.view(out_ch)
            par_idx += bias_size
            # fc layer
            hist_feat = torch.matmul(hist_feat, weight) + bias  # (N, out_ch)
            # activation
            if idx == layers - 1:
                pass
            else:
                hist_feat = self.activation_relu(hist_feat)

        # get module parameters
        assert par_idx == self.total_params - self.module_params
        global_params = params[par_idx]  # (out_ch,)
        global_params = global_params.repeat(N, 1)  # (N, out_ch)
        out_params = global_params + hist_feat  # (N, out_ch), params for each image
        out_params = self.activation_sigmoid(out_params)  # activation, range [0, 1]

        return out_params


# sRGB, 16
class ConditionalGamma(ConditionalModuleBGR):
    def __init__(self, in_channels):
        """
        :param in_channels: (tuple) number of input channels for each layer
        """
        super().__init__(in_channels, 1)  # output channel = 1
        self.kernel = gm.Gamma()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: gamma (self.total_params,)
        :return: output images

        NOTE:
        1. params are not necessarily in range [0, 1]
        2. gamma differs for each individual image (gamma kernel supports this feature)
        """
        # get gamma
        gamma = self._fc_forward(img, params)

        # apply gamma kernel
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        option = 'manual'
        params_dict = {
            'gamma': gamma
            }
        output = self.kernel.run(img, option, params_dict)
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


class WbManual(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = wb.WhiteBalance()

    def forward(self, img, params=None):
        """
        :param img: input images (N, C, H, W)
        :param params: gain (N, 3)
        :return: output images
        """
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)

        # gain, [0, 1] -> [0, 5]
        gain = params * 5

        option = 'manual'
        params_dict = {
            'gain': gain
            }

        output = self.kernel.run(img, option, params_dict)

        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


# sRGB, 17
class ConditionalWbManual(ConditionalModuleBGR):
    def __init__(self, in_channels):
        super().__init__(in_channels, 3)  # 3 parameters
        self.kernel = wb.WhiteBalance()

    def forward(self, img, params=None):
        """
        :param img: input images (N, C, H, W)
        :param params: gain (N, 3)
        :return: output images
        """

        # get gain, in range [0, 5]
        gain = self._fc_forward(img, params) * 5

        # apply wb manual
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        option = 'manual'
        params_dict = {
            'gain': gain
            }
        output = self.kernel.run(img, option, params_dict)
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


class Skip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, params=None):
        # skip connection
        return img


class DemosaicNearest(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = dm.Demosaic()

    def forward(self, img, params=None):
        """
        :param img: input images (N, C, H, W)
        :param params: None
        :return: output images
        """
        # no need to permute

        option = 'nearestneighbor'
        params_dict = {
            'input': {'width': img.shape[3], 'height': img.shape[2], 'format': 'RGGB', 'bitdepth': 10},
            'output': {'width': img.shape[3], 'height': img.shape[2], 'format': 'BGR', 'bitdepth': 8}
        }

        output = self.kernel.run(img, option, params_dict)

        return output


class DemosaicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = dm.Demosaic()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: None
        :return: output images
        """
        # no need to permute

        option = 'demosaicnet'
        params_dict = {
            'input': {'width': img.shape[3], 'height': img.shape[2], 'format': 'RGGB', 'bitdepth': 10},
            'output': {'width': img.shape[3], 'height': img.shape[2], 'format': 'BGR', 'bitdepth': 8}
        }

        output = self.kernel.run(img, option, params_dict)

        return output


class WbQuadratic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: coefficients (N, 30)
        :return: output images
        """
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)

        # params [0, 1] -> [-5, 5]
        params = params * 10 - 5

        # 10 channels (B^2, G^2, R^2, BG, BR, GR, B, G, R, 1)
        ch_b = img[:, :, :, 0:1]
        ch_g = img[:, :, :, 1:2]
        ch_r = img[:, :, :, 2:3]
        ones = torch.ones_like(ch_b)
        img_channels = [ch_b**2, ch_g**2, ch_r**2,
                        ch_b * ch_g, ch_b * ch_r, ch_g * ch_r,
                        ch_b, ch_g, ch_r, ones]
        img_ch = torch.cat(img_channels, dim=3)  # (N, H, W, 10)

        # parameters
        par = params  # (N, 30)
        par = par.view(-1, 1, 3, 10)  # (N, 1, 3, 10)
        par = par.permute(0, 1, 3, 2)  # (N, 1, 10, 3)
        # par = params[0]  # (30,)
        # par = par.view(3, 10)  # (3, 10)
        # par = par.permute(1, 0)  # (10, 3)

        # apply quadratic WB
        # (N, H, W, 10)
        # (N, 1, 10, C), C = 3
        # -------matmul-------
        #  (N, H, W, C)
        img_out = torch.matmul(img_ch, par)

        # permute back to NCHW
        output = img_out.permute(0, 3, 1, 2)  # (N, C, H, W)

        # clip to [0, 1]
        output = torch.clamp(output, 0., 1.)

        return output


# sRGB, 18
class ConditionalWbQuadratic(ConditionalModuleBGR):
    def __init__(self, in_channels):
        super().__init__(in_channels, 30)  # 30 parameters

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: coefficients (N, 30)
        :return: output images
        """
        # get parameters, in range [-5, 5]
        params = self._fc_forward(img, params) * 10 - 5

        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)

        # 10 channels (B^2, G^2, R^2, BG, BR, GR, B, G, R, 1)
        ch_b = img[:, :, :, 0:1]
        ch_g = img[:, :, :, 1:2]
        ch_r = img[:, :, :, 2:3]
        ones = torch.ones_like(ch_b)
        img_channels = [ch_b**2, ch_g**2, ch_r**2,
                        ch_b * ch_g, ch_b * ch_r, ch_g * ch_r,
                        ch_b, ch_g, ch_r, ones]
        img_ch = torch.cat(img_channels, dim=3)  # (N, H, W, 10)

        # parameters
        par = params  # (N, 30)
        par = par.view(-1, 1, 3, 10)  # (N, 1, 3, 10)
        par = par.permute(0, 1, 3, 2)  # (N, 1, 10, 3)

        # apply quadratic WB
        # (N, H, W, 10)
        # (N, 1, 10, C), C = 3
        # -------matmul-------
        #  (N, H, W, C)
        img_out = torch.matmul(img_ch, par)

        # permute back to NCHW
        output = img_out.permute(0, 3, 1, 2)  # (N, C, H, W)

        # clip to [0, 1]
        output = torch.clamp(output, 0., 1.)

        return output


class GtmManual(nn.Module):
    def __init__(self, n_seg):
        super().__init__()
        self.n_seg = n_seg

    def forward(self, imgs, params):
        """
        :param imgs: input images (N, C, H, W)
        :param params: segment points (N, n_seg-1) -- n_seg segments, (n_seg-1) points
        :return:
        """

        # get segment points (the same for the whole batch)
        seg_pts = params[0]

        # apply manual gtm
        output = imgs.clone()
        segment_bound = torch.linspace(0, 1, steps=self.n_seg + 1)
        if imgs.is_cuda:
            segment_bound = segment_bound.cuda()
        for k in range(self.n_seg):
            start_x = segment_bound[k]
            end_x = segment_bound[k+1]
            start_y = seg_pts[k-1] if k > 0 else 0.
            end_y = seg_pts[k] if k < self.n_seg - 1 else 1.
            slope = (end_y - start_y) / (end_x - start_x)
            output = torch.where((imgs >= start_x) & (imgs < end_x), (imgs - start_x) * slope + start_y, output)

        # clip
        output = torch.clamp(output, 0., 1.)

        return output


#---------original tools that have proxy nets-------------
# demosaic, 02
class OriginDemosBilinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = dm.Demosaic()

    def forward(self, img, params=None):
        """
        :param img: input images (N, C, H, W)
        :param params: None
        :return: output images
        """
        # NCHW -> NHWC
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        # [0, 1] -> [0, 255]
        # it does not matter as long as not surpassing 1023 (for this module)
        img = img * 255.

        option = 'bilinear'
        params = { \
            'input': {'width': img.shape[2], 'height': img.shape[1], 'format': 'RGGB', 'bitdepth': 10}, \
            'output': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8}, \
            }

        output = self.kernel.run(img, option, params)

        # [0, 255] -> [0, 1]
        output = output.float() / 255.
        # permute back to NCHW
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


# demosaic, 03
class OriginDemosLaplacian(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = dm.Demosaic()

    def forward(self, img, params=None):
        """
        :param img: input images (N, C, H, W)
        :param params: None
        :return: output images
        """
        # NCHW -> NHWC
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        # [0, 1] -> [0, 255]
        # it does not matter as long as not surpassing 1023 (for this module)
        img = img * 255.

        option = 'laplacian'
        params = { \
            'input': {'width': img.shape[2], 'height': img.shape[1], 'format': 'RGGB', 'bitdepth': 10}, \
            'output': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8}, \
            }

        output = self.kernel.run(img, option, params)

        # [0, 255] -> [0, 1]
        output = output.float() / 255.
        # permute back to NCHW
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


# sRGB, 02
class OriginToneReinhard(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = gtm.GlobalToneMapping()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: (batch, 2), white_point, middle_grey
        :return: output images
        """

        # NCHW -> NHWC
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        # [0, 1] -> [0, 255]
        img = img * 255.

        # parameters
        params = params.detach().cpu().numpy()
        white_point = params[:, 0]
        middle_grey = params[:, 1]

        option = 'reinhard'
        params = {
            'input': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'output': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'white_point': white_point,
            'middle_grey': middle_grey
        }

        output = self.kernel.run(img, option, params)

        # [0, 255] -> [0, 1]
        output = output.float() / 255.
        # permute back to NCHW
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


# sRGB, 03
class OriginToneCrysis(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = gtm.GlobalToneMapping()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: (batch, 1), lum_adapted
        :return: output images
        """
        # NCHW -> NHWC
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        # [0, 1] -> [0, 255]
        img = img * 255.

        # parameters
        params = params.detach().cpu().numpy()
        lum_adapted = params[:, 0]

        option = 'crysisengine'
        params = { \
            'input': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8}, \
            'output': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8}, \
            'lum_adapted': lum_adapted,
            }

        output = self.kernel.run(img, option, params)

        # [0, 255] -> [0, 1]
        output = output.float() / 255.
        # permute back to NCHW
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


# sRGB, 04
class OriginToneFilmic(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = gtm.GlobalToneMapping()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: (batch, 2), white_point, exposure_bias
        :return: output images
        """
        # NCHW -> NHWC
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        # [0, 1] -> [0, 255]
        img = img * 255.

        # parameters
        params = params.detach().cpu().numpy()
        white_point = params[:, 0]
        exposure_bias = params[:, 1]
        # exposure [0, 1] -> [1, 10]
        exposure_bias = exposure_bias * 9. + 1.

        option = 'filmic'
        params = { \
            'input': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8}, \
            'output': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8}, \
            'white_point': white_point,
            'exposure_bias': exposure_bias
            }

        output = self.kernel.run(img, option, params)

        # [0, 255] -> [0, 1]
        output = output.float() / 255.
        # permute back to NCHW
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


# sRGB, 06
class OriginWbWhiteworld(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = wb.WhiteBalance()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: (batch, 1), white point ratio
        :return: output images
        """

        # NCHW -> NHWC
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        # [0, 1] -> [0, 255]
        img = img * 255.

        # parameters
        params = params.detach().cpu().numpy()
        white_point_ratio = params[:, 0]

        option = 'whiteworld'
        params = {
            'input': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'output': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'white_point_ratio': white_point_ratio
            }

        output = self.kernel.run(img, option, params)

        # [0, 255] -> [0, 1]
        output = output.float() / 255.
        # permute back to NCHW
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


# sRGB, 07
class OriginNoiseBilateral(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = snr.SpatialNoiseReduction()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: (batch, 3), window_length, sigma_color, sigma_space; Note: should be torch Tensor
        :return: output images
        """

        # NCHW -> NHWC
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        # [0, 1] -> [0, 255]
        img = img * 255.

        # parameters
        params = params.detach()
        window_length = params[:, 0]
        sigma_color = params[:, 1]
        sigma_space = params[:, 2]

        option = 'bilateral'
        # adjust parameters
        window_length = (window_length.int() * 7) * 2 + 3  # {3, 5, 7, 9, 11, 13, 15}
        sigma_color = sigma_color * 99 + 1  # [1, 100]
        sigma_space = sigma_space * 99 + 1  # [1, 100]

        params = {
            'input': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'output': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'window_length': window_length,
            'sigma_color': sigma_color,
            'sigma_space': sigma_space
            }

        output = self.kernel.run(img, option, params)

        # [0, 255] -> [0, 1]
        output = output.float() / 255.
        # permute back to NCHW
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


# sRGB, 08
class OriginNoiseMedian(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = snr.SpatialNoiseReduction()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: (batch, 1), kernel size
        :return: output images
        """

        # NCHW -> NHWC
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        # [0, 1] -> [0, 255]
        img = img * 255.

        # parameters
        params = params.detach().cpu().numpy()
        kernel_size = params[:, 0]

        option = 'median'
        # [0, 1] -> {3, 5, 7, 9, 11, 13, 15}
        kernel_size = 2 * int(kernel_size[0] * 7) + 3  # unified parameter for the whole batch
        params = {
            'input': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'output': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'size': kernel_size
            }

        output = self.kernel.run(img, option, params)

        # [0, 255] -> [0, 1]
        output = output.float() / 255.
        # permute back to NCHW
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


# sRGB, 09
class OriginNoiseFastnlm(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = snr.SpatialNoiseReduction()

    def forward(self, img, params):
        """
        :param img: input images (N, C, H, W)
        :param params: (batch, 3), block size, search block, decay factor; Note: should be torch Tensor
        :return: output images
        """

        # NCHW -> NHWC
        img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        # [0, 1] -> [0, 255]
        img = img * 255.

        # parameters
        params = params.detach()
        block_size = params[:, 0]
        search_block = params[:, 1]
        decay_factor = params[:, 2]

        option = 'fastnlm'
        block_size = (block_size.int() * 7) * 2 + 3  # {3, 5, 7, 9, 11, 13, 15}
        search_block = (search_block.int() * 7) * 2 + 3  # {3, 5, 7, 9, 11, 13, 15}
        decay_factor = decay_factor * 99 + 1  # [1, 100]
        params = {
            'input': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'output': {'width': img.shape[2], 'height': img.shape[1], 'format': 'BGR', 'bitdepth': 8},
            'block_size': block_size,
            'search_block': search_block,
            'decay_factor': decay_factor
            }

        output = self.kernel.run(img, option, params)

        # [0, 255] -> [0, 1]
        output = output.float() / 255.
        # permute back to NCHW
        output = output.permute(0, 3, 1, 2)  # (N, C, H, W)

        return output


if __name__ == '__main__':
    # GtmManual
    gtm_manual = GtmManual(4)
    gtm_manual.to('cuda')
    # input
    img = np.ones((1, 3, 64, 64), dtype=np.float32) * 0.9  # NCHW
    img_np = img.copy()
    img = torch.from_numpy(img).cuda()
    # parameters
    params = torch.Tensor([[0.3, 0.5, 0.7]]).cuda()
    # output
    out = gtm_manual(img, params)
    out_np = out.detach().cpu().numpy()
    print(out_np.min(), out_np.max())

    # # WbQuadratic
    # wb_quadratic = WbQuadratic()
    # wb_quadratic.to('cuda')
    # # input
    # img = np.ones((1, 3, 64, 64), dtype=np.float32) / 2  # NCHW
    # img_np = img.copy()
    # img = torch.from_numpy(img).cuda()
    # # parameters
    # params = torch.Tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    #                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    #                         0, 0, 0, 0, 0, 0, 0, 0, 1, 0,]]).cuda()
    # # output
    # out = wb_quadratic(img, params)
    # out_np = out.detach().cpu().numpy()
    # print(np.abs(out_np - img_np).mean())

    # # WbManual
    # wb_manual = WbManual()
    # img = np.ones((1, 3, 64, 64), dtype=np.float32) / 2  # NCHW
    # img = torch.from_numpy(img).cuda()
    #
    # # parameters
    # gain = torch.Tensor([[0.2, 0.2, 0.2]]).cuda()  # (1, 3)
    # wb_manual.to('cuda')
    #
    # out = wb_manual(img, gain)
    #

    # # Gamma
    # gamma = Gamma()
    #
    # # parameters
    # ga = torch.Tensor([[0.2]]).cuda()  # (1, 1)
    # gamma.to('cuda')
    #
    # out = gamma(img, ga)
