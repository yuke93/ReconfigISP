import os.path as osp

import numpy as np
import pdb
import torch
import torch.nn as nn

from models.modules.tools_origin import *
from models.modules.tools_proxy import ProxyNet, PathRestore14lBayer, PathRestore14lBgr, ProxyDemosaicNet


class IspUniversal(nn.Module):
    """
    The universal class for proxy ISP.
    """

    def __init__(self, module_path, indiv_module_paths, architecture, **kwargs):
        """
        :param module_path: string, module path
        :param indiv_module_paths: tuple, individual module path for each step, using default path if None
        :param architecture: string, the ISP architecture, example "Bayer_01_Demosaic_03_sRGB_01_13_11"

        Other possible arguments:
        1. gamma_in_channels: (tuple) input channels for conditional gamma
        2. wb_manual_in_channels
        3. wb_quadratic_in_channels
        """

        super().__init__()
        device = torch.device('cuda')

        # default network paths
        model_path = module_path + 'proxy_nets/experiments/'
        # the key is consistent with parameter name
        proxy_dict = {
            'reinhard': (2, model_path + '006_reinhard_residual_multistepLR2/models/400000_G.pth'),
            'crysisengine': (1, model_path + '007_crysis_residual_multistepLR/models/400000_G.pth'),
            'filmic': (2, model_path + '009_filmic_residual_multistepLR/models/400000_G.pth'),
            'whiteworld': (1, model_path + '008_whiteworld_residual_multistepLR/models/400000_G.pth'),
            'bilateral': (3, model_path + '013_bilateral_residual_multistepLR2/models/400000_G.pth'),
            'median': (1, model_path + '010_median_residual_multistepLR/models/400000_G.pth'),
            'fastnlm': (3, model_path + '014_fastnlm_residual_multistepLR2/models/400000_G.pth'),
            # demosaic_bilinear
            'bilinear': (0, model_path + '015_demosaic_bilinear_multistepLR/models/400000_G.pth'),
            # demosaic_laplacian
            'laplacian': (0, model_path + '016_demosaic_laplacian_multistepLR/models/400000_G.pth'),
            # path_restore_14l_bayer
            'path_bayer': (0, model_path + '020_denoise_path_restore_14l_bayer_aug_multistepLR/models/800000_G.pth'),
            # path_restore_14l_bgr
            'path_bgr': (0, model_path + '019_path_restore_14l_rgb/models/path_restore_14l_rgb.pth'),
            'bm3d': (5, model_path + '022_bm3d_residual_multistepLR_mc/models/400000_G.pth')
        }
        # conditional modules (name: in_channels)
        conditional_dict = {
            'conditional_gamma': kwargs['gamma_in_channels'] if 'gamma_in_channels' in kwargs.keys() else None,
            'conditional_wb_manual': kwargs['wb_manual_in_channels'] if 'wb_manual_in_channels' in kwargs.keys() else None,
            'conditional_wb_quadratic': kwargs['wb_quadratic_in_channels'] if 'wb_quadratic_in_channels' in kwargs.keys() else None,
        }

        # algorithm pool, the corresponding names, and default parameter list
        # bayer
        pool_bayer = [PathRestore14lBayer,
                      Skip]
        names_bayer = ['path_bayer', 'skip']
        params_bayer = [[], []]

        pool_demosaic = [DemosaicNearest,
                         ProxyDemosaicNet,
                         ProxyDemosaicNet,
                         DemosaicNet]
        names_demosaic = ['nearest', 'bilinear', 'laplacian', 'demosaicnet']
        params_demosaic = [[], [], [], []]

        pool_srgb = [Gamma,
                     ProxyNet,  # reinhard
                     ProxyNet,  # crysisengine
                     ProxyNet,  # filmic
                     Grayworld,
                     ProxyNet,  # whiteworld
                     ProxyNet,  # bilateral
                     ProxyNet,  # median
                     ProxyNet,  # fastnlm
                     Skip,
                     WbManual,
                     PathRestore14lBgr,
                     WbQuadratic,
                     GtmManual,
                     ProxyNet,  # BM3D
                     ConditionalGamma,  # conditional gamma
                     ConditionalWbManual,  # conditional WB manual
                     ConditionalWbQuadratic,  # conditional WB quadratic
                     TenLayerNet,
                     TwoLayerNet,
                     ToyNet,
                     ]
        names_srgb = ['gamma', 'reinhard', 'crysisengine', 'filmic', 'grayworld',
                      'whiteworld', 'bilateral', 'median', 'fastnlm', 'skip',
                      'wbmanual', 'path_bgr', 'wbquadratic', 'gtmmanual', 'bm3d',
                      'conditional_gamma', 'conditional_wb_manual', 'conditional_wb_quadratic',  # 16-18
                      'ten_layer_net', 'two_layer_net', 'toy_net',  # 19-21
                      ]
        params_srgb = [[0.],  # gamma
                       [0., 0.],  # reinhard
                       [0.],  # crysisengine
                       [0., 0.],  # filmic
                       [],  # grayworld
                       [0.],  # whiteworld
                       [0., 0., 0.],  # bilateral
                       [0.],  # median
                       [0., 0., 0.],  # fastnlm
                       [],  # skip
                       [-1.38, -1.38, -1.38],  # wb manual
                       [],  # path_bgr
                       [0, 0, 0, 0, 0, 0, 0.406, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0.406, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0.406, 0, ],  # wb quadratic
                       [-1.099, 0, 1.099],  # gtm manual
                       [-1.946, 1.099, -1.099, -1.099, 2.708],  # bm3d
                       [0.],  # conditional gamma (global parameters)
                       [-1.38, -1.38, -1.38],  # conditional wb manual (global parameters)
                       [0, 0, 0, 0, 0, 0, 0.406, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0.406, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0.406, 0, ],  # conditional wb quadratic (global parameters)
                       list(np.random.randn(7320) * 0.01),  # ten layer net, hard code init 0.01
                       list(np.random.randn(1763) * 0.01),  # two layer net, hard code init 0.01
                       list(np.random.randn(84) * 0.01),  # toy net, hard code init 0.01
                       ]

        # build ISP
        # architecture is in the form of "Bayer_xx_Demosaic_xx_sRGB_xx_xx_xx"
        architecture_strings = architecture.split('_')
        domain = None
        self.all_modules = []
        self.all_params = []
        self.is_conditional = []  # whether a module is a conditional module
        step = 0
        for arch_str in architecture_strings:
            # check domain
            if arch_str in ['Bayer', 'Demosaic', 'sRGB']:
                domain = arch_str
                continue
            if domain is None:
                raise ValueError('Domain (Bayer, Demosaic, sRGB) is not specified in ISP architecture!')
            step += 1

            if domain == 'Bayer':
                cur_pool = pool_bayer
                cur_names = names_bayer
                cur_params = params_bayer
            elif domain == 'Demosaic':
                cur_pool = pool_demosaic
                cur_names = names_demosaic
                cur_params = params_demosaic
            elif domain == 'sRGB':
                cur_pool = pool_srgb
                cur_names = names_srgb
                cur_params = params_srgb
            else:
                raise ValueError('Invalid domain {}'.format(domain))

            # check the validity of module index
            cur_idx = int(arch_str)
            assert 1 <= cur_idx <= len(cur_pool)
            cur_idx -= 1  # list index starts from zero

            # construct current module
            cur_name = cur_names[cur_idx]
            cur_module_class = cur_pool[cur_idx]
            if cur_name in proxy_dict.keys():  # a (proxy) network
                # check whether to use a specific module path
                indiv_path = indiv_module_paths[step-1]
                if indiv_path is None:
                    cur_module = cur_module_class(*proxy_dict[cur_name]).to(device)
                else:
                    param_num = proxy_dict[cur_name][0]
                    cur_module = cur_module_class(param_num, indiv_path).to(device)
            elif cur_name == 'gtmmanual':
                # hard code, GTM Manual uses 4 segments
                cur_module = cur_module_class(4).to(device)
            elif cur_name in conditional_dict.keys():  # conditional modules
                in_channels = conditional_dict[cur_name]
                assert in_channels is not None
                cur_module = cur_module_class(in_channels=in_channels).to(device)

                # init parameters
                total_params = cur_module.total_params
                global_params = cur_params[cur_idx]
                weight_params = np.random.randn(total_params - len(global_params)) * 0.01  # hard code 0.01 for weight initialization
                weight_params = list(weight_params)
                cur_params[cur_idx] = weight_params + global_params
            else:
                cur_module = cur_module_class().to(device)
            self.all_modules.append(cur_module)

            # update conditional module flag
            if cur_name in conditional_dict.keys():
                self.is_conditional.append(True)
            else:
                self.is_conditional.append(False)

            # construct current parameters
            cur_param_name = 'param_step{}_{}'.format(step, cur_name)
            cur_param = cur_params[cur_idx]
            if len(cur_param) == 0:
                self.all_params.append(nn.Parameter(torch.Tensor([]).to(device)))
            else:
                setattr(self, cur_param_name, nn.Parameter(torch.Tensor(cur_param).to(device)))
                self.all_params.append(getattr(self, cur_param_name))

    def forward(self, x):
        """
        :param x: (N, C, H, W)
        :return: output tensor
        """
        N = x.size()[0]

        self.intermediate_results = []
        for module, param, is_cond in zip(self.all_modules, self.all_params, self.is_conditional):
            if param.nelement() == 0:
                par_tensor = None
            else:
                if is_cond:
                    # no sigmoid activation or repeat for conditional parameters
                    par_tensor = param  # (# of param,)
                else:
                    # valid parameter should be in range [0, 1]
                    par_tensor = torch.sigmoid(param)
                    par_tensor = par_tensor.repeat(N, 1)  # (N, # of param)
            x = module(x, par_tensor)
            self.intermediate_results.append(x)

        return x

    @ property
    def trainable_parameters(self):
        return self.all_params


if __name__ == '__main__':
    device = 'cuda:0'
    isp_net = IspUniversal(module_path='/DATA/module/',
                           indiv_module_paths=(None, 'not used', None, None, None, None),
                           architecture='Bayer_01_02_Demosaic_04_sRGB_01_13_11')
    isp_net = isp_net.to(device)
    x = torch.ones(16, 1, 64, 64) / 2.
    x = x.to(device)
    y = isp_net(x)
    # pdb.set_trace()
