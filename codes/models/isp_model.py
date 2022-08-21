import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
import copy
import pdb

logger = logging.getLogger('base')


class IspModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # not use distributed training
        # todo: distributed training
        self.rank = -1
        self.netG = networks.define_G(opt).to(self.device)
        self.netG_attr = self.netG
        # self.netG = DataParallel(self.netG)
        # self.netV = copy.deepcopy(self.netG)
        # self.netV = networks.define_G(opt).to(self.device)  # build another net instead of deep copy
        self.print_network()
        self.load()
        self.img = None  # input images
        self.output = None  # output images
        self.gt = None  # ground truth
        self.val_img = None
        self.val_gt = None
        self.l_pix = None  # loss
        self.meta = None  # meta data that implies the camera device

        # training
        if self.is_train:
            train_opt = opt['train']
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.cri_pix_v = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_pix_v = nn.MSELoss().to(self.device)

            # get hyper-parameters for optimizer
            lr_G = train_opt['lr_G']
            beta1 = train_opt['beta1']
            beta2 = train_opt['beta2']

            # optimizer for parameters
            # self.optimizer_G = torch.optim.SGD(self.netG.trainable_parameters, lr_G, momentum=self.momentum_G)
            self.optimizer_G = torch.optim.Adam(self.netG.trainable_parameters, lr_G, (beta1, beta2))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

        else:
            self.netG.eval()

        self.log_dict = OrderedDict()

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def get_current_log(self):
        return self.log_dict

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def feed_data(self, data):
        """data contains images and GT"""
        if len(data) == 2:
            img, gt = data
        elif len(data) == 3:
            img, gt, meta = data
            self.meta = meta.to(self.device)
        elif len(data) == 4:
            img, gt, val_img, val_gt = data
            self.val_img = val_img.to(self.device)
            self.val_gt = val_gt.to(self.device)
        elif len(data) == 5:
            img, gt, val_img, val_gt, meta = data
            self.val_img = val_img.to(self.device)
            self.val_gt = val_gt.to(self.device)
            self.meta = meta.to(self.device)
        else:
            raise ValueError('Invalid data format.')
        self.img = img.to(self.device)
        self.gt = gt.to(self.device)

    def optimize_parameters(self):
        # loss function
        if self.meta is None:
            self.output = self.netG(self.img)
        else:
            self.output = self.netG(self.img, self.meta)
        self.l_pix = self.cri_pix(self.output, self.gt)

        # backward
        self.optimizer_G.zero_grad()
        self.l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['loss'] = self.l_pix.item()

    def test(self):
        if self.meta is None:
            self.output = self.netG(self.img)
        else:
            self.output = self.netG(self.img, self.meta)
        intermediate_results = self.netG.intermediate_results

        return self.output, intermediate_results
