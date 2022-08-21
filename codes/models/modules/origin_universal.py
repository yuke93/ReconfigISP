import pdb
import torch
import torch.nn as nn

from models.modules.tools_origin import *
from models.modules.tools_proxy import ProxyNet, PathRestore14lBayer, PathRestore14lBgr


class OriginUniversal(nn.Module):
    """
    The universal class for origin ISP.
    Only BM3D is a proxy network since the original algorithm is implemented in MATLAB
    """

    def __init__(self, module_path, architecture):
        """
        :param module_path: string, module path (for BM3D proxy, DemosaicNet, Path-Restore-Bayer and Path-Restore)
        :param architecture: string, the ISP architecture, example "Bayer_01_Demosaic_03_sRGB_01_13_11"
        """

        super().__init__()
        device = torch.device('cuda')

        # (proxy) networks
        # only BM3D is a proxy network, the others are deep methods
        proxy_dict = {
            'bm3d': (5, module_path + 'proxy_nets/experiments/022_bm3d_residual_multistepLR_mc/models/400000_G.pth'),
            'path_restore_14l_bayer': (0, module_path +
                'proxy_nets/experiments/020_denoise_path_restore_14l_bayer_aug_multistepLR/models/800000_G.pth'),
            'path_restore_14l_bgr': (0, module_path +
                'proxy_nets/experiments/019_path_restore_14l_rgb/models/path_restore_14l_rgb.pth'),
        }

        # algorithm pool, the corresponding names, and default parameter list
        # bayer
        pool_bayer = [PathRestore14lBayer,
                      Skip]
        names_bayer = ['path_bayer', 'skip']
        params_bayer = [[], []]

        pool_demosaic = [DemosaicNearest,
                         OriginDemosBilinear,
                         OriginDemosLaplacian,
                         DemosaicNet]
        names_demosaic = ['nearest', 'bilinear', 'laplacian', 'demosaicnet']
        params_demosaic = [[], [], [], []]

        pool_srgb = [Gamma,
                     OriginToneReinhard,
                     OriginToneCrysis,
                     OriginToneFilmic,
                     Grayworld,
                     OriginWbWhiteworld,
                     OriginNoiseBilateral,
                     OriginNoiseMedian,
                     OriginNoiseFastnlm,
                     Skip,
                     WbManual,
                     PathRestore14lBgr,
                     WbQuadratic,
                     GtmManual(4).to(device),
                     ProxyNet]
        names_srgb = ['gamma', 'reinhard', 'crysisengine', 'filmic', 'grayworld',
                      'whiteworld', 'bilateral', 'median', 'fastnlm', 'skip',
                      'wbmanual', 'path_bgr', 'wbquadratic', 'gtmmanual', 'bm3d']
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
                       ]

        # build ISP
        # architecture is in the form of "Bayer_xx_Demosaic_xx_sRGB_xx_xx_xx"
        architecture_strings = architecture.split('_')
        domain = None
        self.all_modules = []
        self.all_params = []
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
            if cur_name == 'path_bayer':
                cur_module = cur_module_class(*proxy_dict['path_restore_14l_bayer']).to(device)
            elif cur_name == 'path_bgr':
                cur_module = cur_module_class(*proxy_dict['path_restore_14l_bgr']).to(device)
            elif cur_name == 'bm3d':
                cur_module = cur_module_class(*proxy_dict['bm3d']).to(device)
            else:
                cur_module = cur_module_class().to(device)
            self.all_modules.append(cur_module)

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
        for module, param in zip(self.all_modules, self.all_params):
            if param.nelement() == 0:
                par_tensor = None
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
    isp_net = OriginUniversal(module_path='/DATA/module/',
                              architecture='Bayer_01_02_Demosaic_04_sRGB_01_13_11_05_02_03')
    isp_net = isp_net.to(device)
    x = torch.ones(16, 1, 64, 64) / 2.
    x = x.to(device)
    y = isp_net(x)
    pdb.set_trace()
