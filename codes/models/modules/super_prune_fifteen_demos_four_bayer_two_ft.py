import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.tools_origin import Grayworld, Gamma, Skip, WbManual, DemosaicNearest, DemosaicNet, WbQuadratic
from models.modules.tools_origin import GtmManual
from models.modules.tools_proxy import ProxyNet, ProxyDemosaicNet, PathRestore14lBayer, PathRestore14lBgr
import pdb


# A super net that contains 15 modules with online pruning (include manual global tone mapping and BM3D),
# and 4 demosaicking methods (not counted as one step),
# and 2 Bayer methods -- 14-layer Path-Restore and skip connection (not counted as one step)
# Allow proxy fine-tuning
class SuperPruneFifteenDemosFourBayerTwoFt(nn.Module):
    def __init__(self, n_step, threshold, module_path):
        # at each step, it requires a list of modules,
        # a list of parameters (as the module input),
        # and a list of alphas (module selection/combination)
        """
        :param n_step: number of steps (total # of modules in an ISP pipeline)
        :param threshold: threshold ratio for pruning
        :param module_path: module path (for proxy nets)
        """

        super().__init__()
        self.threshold = threshold

        # intermediate results
        self.middle_results = None

        # number of pruned paths at each step
        self.pruned_paths = [0] * (n_step + 2)  # (include demosaic, bayer)

        device = torch.device('cuda')
        proxy_dict = {
            'reinhard': (2, module_path + 'proxy_nets/experiments/006_reinhard_residual_multistepLR2/models/400000_G.pth'),
            'crysisengine': (1, module_path + 'proxy_nets/experiments/007_crysis_residual_multistepLR/models/400000_G.pth'),
            'filmic': (2, module_path + 'proxy_nets/experiments/009_filmic_residual_multistepLR/models/400000_G.pth'),
            'whiteworld': (1, module_path + 'proxy_nets/experiments/008_whiteworld_residual_multistepLR/models/400000_G.pth'),
            'bilateral': (3, module_path + 'proxy_nets/experiments/013_bilateral_residual_multistepLR2/models/400000_G.pth'),
            'median': (1, module_path + 'proxy_nets/experiments/010_median_residual_multistepLR/models/400000_G.pth'),
            'fastnlm': (3, module_path + 'proxy_nets/experiments/014_fastnlm_residual_multistepLR2/models/400000_G.pth'),
            'demosaic_bilinear': (0, module_path + 'proxy_nets/experiments/015_demosaic_bilinear_multistepLR/models/400000_G.pth'),
            'demosaic_laplacian': (0, module_path + 'proxy_nets/experiments/016_demosaic_laplacian_multistepLR/models/400000_G.pth'),
            'path_restore_14l_bayer': (0, module_path + 'proxy_nets/experiments/020_denoise_path_restore_14l_bayer_aug_multistepLR/models/800000_G.pth'),
            'path_restore_14l_bgr': (0, module_path + 'proxy_nets/experiments/019_path_restore_14l_rgb/models/path_restore_14l_rgb.pth'),
            'bm3d': (5, module_path + 'proxy_nets/experiments/022_bm3d_residual_multistepLR_mc/models/400000_G.pth')
        }

        # modules, params, alphas
        self.all_modules = []
        self.all_params = []
        self.all_alphas = []
        self.trainable_params = []
        self.param_and_alpha = []

        # Bayer step
        cur_modules = [
            PathRestore14lBayer(*proxy_dict['path_restore_14l_bayer']).to(device),
            Skip().to(device)
        ]

        cur_params = [
            nn.Parameter(torch.Tensor([]).to(device)),
            nn.Parameter(torch.Tensor([]).to(device))
        ]

        setattr(self, 'alpha_bayer', nn.Parameter(torch.zeros((2,)).to(device)))
        cur_alphas = getattr(self, 'alpha_bayer')

        cur_modules = nn.ModuleList(cur_modules)
        cur_params = nn.ParameterList(cur_params)
        self.all_modules.append(cur_modules)
        self.all_params.append(cur_params)
        self.all_alphas.append(cur_alphas)

        # demosaic step
        cur_modules = [
            DemosaicNearest().to(device),
            ProxyDemosaicNet(*proxy_dict['demosaic_bilinear']).to(device),
            ProxyDemosaicNet(*proxy_dict['demosaic_laplacian']).to(device),
            DemosaicNet().to(device),
        ]

        cur_params = [
            nn.Parameter(torch.Tensor([]).to(device)),
            nn.Parameter(torch.Tensor([]).to(device)),
            nn.Parameter(torch.Tensor([]).to(device)),
            nn.Parameter(torch.Tensor([]).to(device))
        ]

        setattr(self, 'alpha_demosaic', nn.Parameter(torch.zeros((4,)).to(device)))
        cur_alphas = getattr(self, 'alpha_demosaic')

        cur_modules = nn.ModuleList(cur_modules)
        cur_params = nn.ParameterList(cur_params)
        self.all_modules.append(cur_modules)
        self.all_params.append(cur_params)
        self.all_alphas.append(cur_alphas)

        # BGR steps
        # proxy list (name, flag indicating whether proxy fine-tuning is enabled)
        self.proxy_ft_flag = [('gamma', 0),
                              ('reinhard', 0),  # need to fine-tune, but has nan bug
                              ('crysisengine', 1),
                              ('filmic', 0),  # need to fine-tune, but has nan bug
                              ('grayworld', 0),
                              ('whiteworld', 1),
                              ('bilateral', 1),
                              ('median', 1),
                              ('fastnlm', 1),
                              ('skip', 0),
                              ('wbmanual', 0),
                              ('path_restore_14l_bgr', 0),
                              ('wbquadratic', 0),
                              ('gtmmanual', 0),
                              ('bm3d', 0),  # should be fine-tuned, but cannot do it
                              ]
        self.n_step = n_step  # need this parameter
        for k in range(n_step):
            cur_modules = [
                Gamma().to(device),
                ProxyNet(*proxy_dict['reinhard']).to(device),
                ProxyNet(*proxy_dict['crysisengine']).to(device),
                ProxyNet(*proxy_dict['filmic']).to(device),
                Grayworld().to(device),
                ProxyNet(*proxy_dict['whiteworld']).to(device),
                ProxyNet(*proxy_dict['bilateral']).to(device),
                ProxyNet(*proxy_dict['median']).to(device),
                ProxyNet(*proxy_dict['fastnlm']).to(device),
                Skip().to(device),
                WbManual().to(device),
                PathRestore14lBgr(*proxy_dict['path_restore_14l_bgr']).to(device),
                WbQuadratic().to(device),
                GtmManual(4).to(device),
                ProxyNet(*proxy_dict['bm3d']).to(device)
            ]

            # set parameters (add to state_dict)
            setattr(self, 'param_step{}_gamma'.format(k + 1), nn.Parameter(torch.Tensor([0.]).to(device)))
            setattr(self, 'param_step{}_reinhard'.format(k + 1), nn.Parameter(torch.Tensor([0., 0.]).to(device)))
            setattr(self, 'param_step{}_crysisengine'.format(k + 1), nn.Parameter(torch.Tensor([0.]).to(device)))
            setattr(self, 'param_step{}_filmic'.format(k + 1), nn.Parameter(torch.Tensor([0., 0.]).to(device)))
            # no param for Grayworld
            setattr(self, 'param_step{}_whiteworld'.format(k + 1), nn.Parameter(torch.Tensor([0.]).to(device)))
            setattr(self, 'param_step{}_bilateral'.format(k + 1), nn.Parameter(torch.Tensor([0., 0., 0.]).to(device)))
            setattr(self, 'param_step{}_median'.format(k + 1), nn.Parameter(torch.Tensor([0.]).to(device)))
            setattr(self, 'param_step{}_fastnlm'.format(k + 1), nn.Parameter(torch.Tensor([0., 0., 0.]).to(device)))
            # no param for Skip
            setattr(self, 'param_step{}_wbmanual'.format(k + 1), nn.Parameter(torch.Tensor([-1.38, -1.38, -1.38]).to(device)))
            # no param for PathRestore14lBgr
            setattr(self, 'param_step{}_wbquadratic'.format(k + 1),
                    nn.Parameter(torch.Tensor([0, 0, 0, 0, 0, 0, 0.406, 0,     0,     0,
                                               0, 0, 0, 0, 0, 0, 0,     0.406, 0,     0,
                                               0, 0, 0, 0, 0, 0, 0,     0,     0.406, 0,]).to(device)))
            setattr(self, 'param_step{}_gtmmanual'.format(k + 1), nn.Parameter(torch.Tensor([-1.099, 0, 1.099]).to(device)))
            # BM3D params: cff, n1, cspace, wtransform, neighborhood; init value: 1.5, 8, 0, 0, 8
            # init probs: 0.125, 0.75, 0.25, 0.25, 0.9375; init logits: -1.946, 1.099, -1.099, -1.099, 2.708
            setattr(self, 'param_step{}_bm3d'.format(k + 1), nn.Parameter(torch.Tensor([-1.946, 1.099, -1.099, -1.099, 2.708]).to(device)))

            cur_params = [
                getattr(self, 'param_step{}_gamma'.format(k + 1)),
                getattr(self, 'param_step{}_reinhard'.format(k + 1)),
                getattr(self, 'param_step{}_crysisengine'.format(k + 1)),
                getattr(self, 'param_step{}_filmic'.format(k + 1)),
                nn.Parameter(torch.Tensor([]).to(device)),
                getattr(self, 'param_step{}_whiteworld'.format(k + 1)),
                getattr(self, 'param_step{}_bilateral'.format(k + 1)),
                getattr(self, 'param_step{}_median'.format(k + 1)),
                getattr(self, 'param_step{}_fastnlm'.format(k + 1)),
                nn.Parameter(torch.Tensor([]).to(device)),
                getattr(self, 'param_step{}_wbmanual'.format(k + 1)),
                nn.Parameter(torch.Tensor([]).to(device)),
                getattr(self, 'param_step{}_wbquadratic'.format(k + 1)),
                getattr(self, 'param_step{}_gtmmanual'.format(k + 1)),
                getattr(self, 'param_step{}_bm3d'.format(k + 1)),
            ]

            # alphas
            setattr(self, 'alpha_step{}'.format(k + 1), nn.Parameter(torch.zeros((15,)).to(device)))  # 15 BGR modules
            cur_alphas = getattr(self, 'alpha_step{}'.format(k + 1))

            self.trainable_params += cur_params

            cur_modules = nn.ModuleList(cur_modules)
            cur_params = nn.ParameterList(cur_params)

            self.all_modules.append(cur_modules)
            self.all_params.append(cur_params)
            self.all_alphas.append(cur_alphas)

        self.param_and_alpha = self.trainable_params + self.all_alphas

    def load_proxy_nets(self, name_net_dict):
        """ Update the fine-tuned proxy network weights
        :param name_net_dict: a dict {name: target net}
        :return:
        """
        for idx, (name, ft_flag) in enumerate(self.proxy_ft_flag):
            if ft_flag:
                target_net = name_net_dict[name]
                target_state_dict = target_net.state_dict()
                # print('Update fine-tuned weights of {}'.format(name))  # log info
                # copy the weights of target net to current nets
                for k in range(self.n_step):
                    # target net and current net should be the same when k==0
                    # but still load for safety
                    cur_net = self.all_modules[-1-k][idx]
                    cur_net.load_state_dict(target_state_dict)

    def forward(self, x):
        """
        :param x: (N, 1, H, W), bayer pattern
        :return: output tensor
        """
        N = x.size()[0]

        self.middle_results = []
        for step_idx, (module, param, alpha) in enumerate(zip(self.all_modules, self.all_params, self.all_alphas)):
            y = 0
            probs = F.softmax(alpha, dim=0)

            # prune the probs
            max_prob = probs.max()
            post_probs = probs.clone()
            probs = probs.detach()  # no grad
            post_probs[probs < self.threshold * max_prob] = 0
            post_probs = post_probs / post_probs.sum().detach()  # normalize
            self.pruned_paths[step_idx] = (probs < self.threshold * max_prob).sum().item()

            for mod, par, prob in zip(module, param, post_probs):
                # continue if prob == 0
                if prob < 1e-9:
                    # dummy gradients for par
                    if par.nelement() > 0:
                        y += torch.zeros(x.shape).cuda() * par.sum()
                    continue

                # repeat parameters
                if par.nelement() == 0:
                    par_tensor = None
                else:
                    # valid parameter should be in range [0, 1]
                    par_tensor = torch.sigmoid(par)
                    par_tensor = par_tensor.repeat(N, 1)  # (N, # of param)
                y = y + mod(x, par_tensor) * prob
            self.middle_results.append(y)
            x = y  # output feature at current step / input tensor at next step

        return x

    @ property
    def trainable_parameters(self):
        return self.trainable_params

    @ property
    def parameters_and_alpha(self):
        return self.param_and_alpha

    @ property
    def alphas(self):
        return self.all_alphas

    @ property
    def intermediate_results(self):
        return self.middle_results


if __name__ == '__main__':
    # investigate proxy fine-tuning
    module_path = '/DATA/module/'
    netG = SuperPruneFifteenDemosFourBayerTwoFt(n_step=2, threshold=0.5, module_path=module_path)
    pdb.set_trace() 
