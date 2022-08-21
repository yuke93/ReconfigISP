# tools -- proxy modules
# tone mapping: reinhard, crysisengine, filmic, whiteworld
# white balance: whiteworld

import torch
import torch.nn as nn
from models.modules.srcnn_res_arch import SRCNNRes
from models.modules.srcnn_demosaic_arch import SRCNNDemosaic
from models.modules.path_14l_bayer_arch import Path14lBayer
from models.modules.path_14l_bgr_arch import Path14lBgr
import logging
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
import pdb


class ProxyNet(SRCNNRes):
    def __init__(self, param_channel, load_path, strict_load=True):
        """
        :param param_channel: number of parameters
        :param load_path: load path
        """
        super().__init__(param_channel)
        self.logger = logging.getLogger('base')
        if load_path is not None:
            self.load(load_path, strict_load)

    def load(self, load_path, strict_load):
        network = self
        self.logger.info('Loading model for ProxyNet [{:s}] ...'.format(load_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict_load)


class ProxyDemosaicNet(SRCNNDemosaic):
    def __init__(self, param_channel, load_path, strict_load=True):
        """
        :param param_channel: number of parameters
        :param load_path: load path
        """
        super().__init__(param_channel)
        self.logger = logging.getLogger('base')
        if load_path is not None:
            self.load(load_path, strict_load)

    def load(self, load_path, strict_load):
        network = self
        self.logger.info('Loading model for ProxyNet [{:s}] ...'.format(load_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict_load)


class PathRestore14lBayer(Path14lBayer):
    def __init__(self, param_channel, load_path, strict_load=True):
        """
        :param param_channel: number of parameters
        :param load_path: load path
        """
        super().__init__(param_channel)
        self.logger = logging.getLogger('base')
        if load_path is not None:
            self.load(load_path, strict_load)

    def load(self, load_path, strict_load):
        network = self
        self.logger.info('Loading model for ProxyNet [{:s}] ...'.format(load_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict_load)


class PathRestore14lBgr(Path14lBgr):
    def __init__(self, param_channel, load_path, strict_load=True):
        """
        :param param_channel: number of parameters
        :param load_path: load path
        """
        super().__init__(param_channel)
        self.logger = logging.getLogger('base')
        if load_path is not None:
            self.load(load_path, strict_load)

    def load(self, load_path, strict_load):
        network = self
        self.logger.info('Loading model for ProxyNet [{:s}] ...'.format(load_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict_load)
