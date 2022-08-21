import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
import torch.distributed as dist
from models.modules.tools_origin import OriginToneReinhard, OriginToneCrysis, OriginToneFilmic, \
    OriginWbWhiteworld, OriginNoiseBilateral, OriginNoiseMedian, OriginNoiseFastnlm
import copy
import pdb
import numpy as np
import random

logger = logging.getLogger('base')


class DartsFtModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        if opt['dist']:
            self.rank = dist.get_rank()
        else:
            self.rank = -1
        self.netG = networks.define_G(opt).to(self.device)
        # build another net instead of deep copy
        self.netV = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.netG_attr = self.netG.module
        else:
            self.netG_attr = self.netG

        if self.rank <= 0:
            self.print_network()
        self.load()
        self.img = None  # input images
        self.output = None  # output images
        self.gt = None  # ground truth
        self.val_img = None
        self.val_gt = None
        self.val_loss = None  # train valid loss

        # training
        if self.is_train:
            train_opt = opt['train']
            self.netG.train()

            ## settings for proxy fine-tuning
            # settings
            self.memory_size = opt['proxy_ft_params']['memory_size']
            self.ft_steps = opt['proxy_ft_params']['ft_steps']

            # data_memory
            self.ft_data = []  # data memory for proxy fine-tuning

            # number of parameter dict
            self.param_num_dict = {'reinhard': 2,
                                   'crysisengine': 1,
                                   'filmic': 2,
                                   'whiteworld': 1,
                                   'bilateral': 3,
                                   'median': 1,
                                   'fastnlm': 3}

            # set target dict {name: target_net}
            target_dict = {'reinhard': OriginToneReinhard(),
                           'crysisengine': OriginToneCrysis(),
                           'filmic': OriginToneFilmic(),
                           'whiteworld': OriginWbWhiteworld(),
                           'bilateral': OriginNoiseBilateral(),
                           'median': OriginNoiseMedian(),
                           'fastnlm': OriginNoiseFastnlm()}
            # construct a list of modules that need fine-tuning
            # each element is [name, net_proxy, net_proxy_attr, net_target, optimizer]
            self.ft_nets = []
            for (name, ft_flag), net_proxy in zip(self.netG_attr.proxy_ft_flag, self.netG_attr.all_modules[-1]):
                if ft_flag:
                    if opt['dist']:
                        cur_net = DistributedDataParallel(net_proxy, device_ids=[torch.cuda.current_device()])
                        cur_net_attr = cur_net.module
                    else:
                        cur_net = net_proxy
                        cur_net_attr = net_proxy
                    cur_net.train()

                    # optimizer
                    # todo: may adjust the training scheme
                    cur_optimizer = torch.optim.Adam(cur_net_attr.parameters(), lr=train_opt['lr_G'],
                                                     betas=(train_opt['beta1'], train_opt['beta2']))

                    cur_name_nets = [name, cur_net, cur_net_attr, target_dict[name], cur_optimizer]
                    self.ft_nets.append(cur_name_nets)

            # normal training settings of darts
            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.cri_pix_v = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_pix_v = nn.MSELoss().to(self.device)

            # get hyper-parameters for SGD op
            lr_G = train_opt['lr_G']
            self.momentum_G = train_opt['momentum_G']
            # lr for meta learning
            self.lr_meta = train_opt['lr_meta']

            # optimizer for parameters
            self.optimizer_G = torch.optim.SGD(self.netG_attr.trainable_parameters, lr_G, momentum=self.momentum_G)
            self.optimizers.append(self.optimizer_G)

            # optimizer for alphas
            self.optimizer_alpha = torch.optim.Adam(self.netG_attr.alphas, lr=train_opt['lr_G'],
                                                    betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_alpha)

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
        # save fine-tuned models
        for (name, _, proxy_attr, _, _) in self.ft_nets:
            self.save_network(proxy_attr, name, iter_label)

    def feed_data(self, data):
        """data contains images and GT"""
        if len(data) == 2:
            img, gt = data
        elif len(data) == 4:
            img, gt, val_img, val_gt = data
            self.val_img = val_img.to(self.device)
            self.val_gt = val_gt.to(self.device)
        else:
            raise ValueError('Invalid data format.')
        self.img = img.to(self.device)
        self.gt = gt.to(self.device)

    def optimize_parameters(self):
        # loss function
        self.output = self.netG(self.img)
        l_pix = self.cri_pix(self.output, self.gt)

        # backward
        self.optimizer_G.zero_grad()
        l_pix.backward()
        self.optimizer_G.step()

        # update data memory for proxy finetuning
        # a queue (first in first out)
        middle_results = [tensor.clone().detach().cpu() for tensor in self.netG_attr.intermediate_results
                          if tensor.size()[1] == 3]  # only consider sRGB value
        self.ft_data.extend(middle_results)
        if len(self.ft_data) > self.memory_size:
            crop_index = len(self.ft_data) - self.memory_size
            self.ft_data = self.ft_data[crop_index:]

        # set log
        self.log_dict['loss'] = l_pix.item()

    def finetune_proxies(self):
        """ Fine-tune and update proxy nets """
        if not self.is_train:
            return

        # fine-tune each valid proxy net
        name_net_dict = {}
        for (name, proxy, proxy_attr, target_net, optimizer) in self.ft_nets:
            # sample data
            data_len = len(self.ft_data)
            if data_len == 0:
                print('[Warning] Data is not ready for proxy fine-tuning!')
                continue

            # fine-tune several steps at once
            for _ in range(self.ft_steps):
                data_idx = int(random.random() * data_len)
                data = self.ft_data[data_idx].to(self.device)

                # sample parameters
                param_num = self.param_num_dict[name]
                cur_batch = data.size()[0]
                param = torch.rand(1, param_num).repeat(cur_batch, 1).to(self.device)  # cur_batch x param_num

                # proxy net inference
                output = proxy(data, param)  # use proxy with data parallel

                # target inference
                gt = target_net(data, param)

                # optimization (use the same cri_pix)
                loss = self.cri_pix(output, gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update name_net_dict
            name_net_dict.update({name: proxy_attr})  # Note proxy_attr should also be updated when optimizing proxy

        # update the weights of proxy nets
        self.netG_attr.load_proxy_nets(name_net_dict)

    def virtual_step(self):
        """
        Compute unrolled parameters p' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        """
        # forward & calc loss
        output = self.netG(self.img)
        loss = self.cri_pix(output, self.gt)

        # compute gradients
        gradients = torch.autograd.grad(loss, self.netG_attr.trainable_parameters, allow_unused=True)

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            # parameter, virtual parameter, gradient
            for p, vp, g in zip(self.netG_attr.trainable_parameters, self.netV.trainable_parameters, gradients):
                momentum = self.optimizer_G.state[p].get('momentum_buffer', 0.) * self.momentum_G
                if len(p) > 0:
                    if g is None:
                        vp.copy_(p)
                    else:
                        vp.copy_(p - self.lr_meta * (momentum + g))

            # synchronize alphas
            for a, va in zip(self.netG_attr.alphas, self.netV.alphas):
                va.copy_(a)

    def optimize_alphas(self):
        """ Compute unrolled loss and backward its gradients """
        # zero grad
        self.optimizer_alpha.zero_grad()

        # do virtual step (calc p`)
        self.virtual_step()

        # calc unrolled loss (on val data)
        val_output = self.netV(self.val_img)
        loss = self.cri_pix_v(val_output, self.val_gt)  # L_val(p`)
        self.val_loss = loss

        # compute gradients
        v_alphas = tuple(self.netV.alphas)
        v_params = tuple(self.netV.trainable_parameters)
        v_grads = torch.autograd.grad(loss, v_alphas + v_params, allow_unused=True)
        dalpha = v_grads[:len(v_alphas)]
        dp = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dp)

        # update final gradient = dalpha - lr_meta * hessian
        with torch.no_grad():
            idx = 0
            for alpha, da, h in zip(self.netG_attr.alphas, dalpha, hessian):
                idx += 1
                if da is None or h is None:
                    alpha.grad = torch.zeros(alpha.shape).cuda()
                elif torch.isnan(h).any():
                    print('Warning: NaN in hessian, for the {}-th alpha'.format(idx))
                    # pdb.set_trace()
                    alpha.grad = torch.zeros(alpha.shape).cuda()
                else:
                    alpha.grad = da - self.lr_meta * h

        # apply gradients
        self.optimizer_alpha.step()

    def compute_hessian(self, dp):
        """
        dp = dp` { L_val(p`, alpha) }
        p+ = p + eps * dp
        p- = p - eps * dp
        hessian = (dalpha { L_trn(p+, alpha) } - dalpha { L_trn(p-, alpha) }) / (2*eps)
        eps = 0.01 / ||dp||
        """
        norm = torch.cat([w.view(-1) for w in dp if w is not None]).norm()
        # if norm is too small, return dummy hessian with all 0
        if norm < 1e-6:
            eps = 0.
        else:
            eps = 0.01 / norm

        # p+ = p + eps*dp`
        with torch.no_grad():
            for p, d in zip(self.netG_attr.trainable_params, dp):
                if len(p) > 0 and d is not None:
                    p += eps * d
        output = self.netG(self.img)
        loss = self.cri_pix(output, self.gt)
        dalpha_pos = torch.autograd.grad(loss, self.netG_attr.alphas)  # dalpha { L_trn(p+) }

        # p- = p - eps*dp`
        with torch.no_grad():
            for p, d in zip(self.netG_attr.trainable_params, dp):
                if len(p) > 0 and d is not None:
                    p -= 2. * eps * d
        output = self.netG(self.img)
        loss = self.cri_pix(output, self.gt)
        dalpha_neg = torch.autograd.grad(loss, self.netG_attr.alphas)  # dalpha { L_trn(p-) }

        # recover p
        with torch.no_grad():
            for p, d in zip(self.netG_attr.trainable_params, dp):
                if len(p) > 0 and d is not None:
                    p += eps * d

        hessian = [(p - n) / 2. * eps if p is not None and n is not None else None for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def test(self):
        self.output = self.netG(self.img)
        intermediate_results = self.netG_attr.intermediate_results

        return self.output, intermediate_results
