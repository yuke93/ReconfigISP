import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
import torch.distributed as dist
import copy
import pdb
import numpy as np
from utils.util_loss import local_global_loss, latency_loss
from functools import partial

logger = logging.getLogger('base')


class DartsModel(BaseModel):
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
        self.glb_flag = None  # global flag
        self.val_img = None
        self.val_gt = None
        self.val_loss = None  # train valid loss
        self.val_glb_flag = None
        # latency (update when optimizing parameters)
        self.latency = None
        self.latency_term = None

        # training
        if self.is_train:
            train_opt = opt['train']
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            self.is_local_global = 'local_global' in loss_type  # whether to use local_global loss
            self.is_latency = 'latency' in loss_type
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.cri_pix_v = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_pix_v = nn.MSELoss().to(self.device)
            elif loss_type == 'local_global_l2':
                mse_func = nn.MSELoss().to(self.device)
                mse_func_v = nn.MSELoss().to(self.device)
                self.cri_pix = partial(local_global_loss, loss_func=mse_func)
                self.cri_pix_v = partial(local_global_loss, loss_func=mse_func_v)
            elif loss_type == 'l2_latency':
                mse_func = nn.MSELoss().to(self.device)
                mse_func_v = nn.MSELoss().to(self.device)
                param_w = train_opt['w']
                target_latency = train_opt['target_latency']
                self.cri_pix = partial(latency_loss, target_latency=target_latency, w=param_w, fidelity_loss=mse_func)
                self.cri_pix_v = partial(latency_loss, target_latency=target_latency, w=param_w, fidelity_loss=mse_func_v)

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

    def feed_data(self, data):
        """data contains images and GT"""
        if len(data) == 2:
            img, gt = data
        elif len(data) == 4:
            img, gt, val_img, val_gt = data
            self.val_img = val_img.to(self.device)
            self.val_gt = val_gt.to(self.device)
        elif len(data) == 6:
            # local global loss
            img, gt, glb_flag, val_img, val_gt, val_glb_flag = data
            self.glb_flag = glb_flag.to(self.device)
            self.val_img = val_img.to(self.device)
            self.val_gt = val_gt.to(self.device)
            self.val_glb_flag = val_glb_flag.to(self.device)
        else:
            raise ValueError('Invalid data format.')
        self.img = img.to(self.device)
        self.gt = gt.to(self.device)

    def optimize_parameters(self):
        # loss function
        if self.is_latency:
            self.output, self.latency = self.netG(self.img)
            l_pix, self.latency_term = self.cri_pix(self.output, self.gt, self.latency)
        else:
            self.output = self.netG(self.img)
            if self.is_local_global:
                l_pix = self.cri_pix(self.output, self.gt, self.glb_flag)
            else:
                l_pix = self.cri_pix(self.output, self.gt)

        # backward
        self.optimizer_G.zero_grad()
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['loss'] = l_pix.item()
        if self.is_latency:
            self.log_dict['latency'] = self.latency.item()
            self.log_dict['latency_term'] = self.latency_term.item()

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
        if self.is_latency:
            output, latency = self.netG(self.img)
            loss, _ = self.cri_pix(output, self.gt, latency)
        else:
            output = self.netG(self.img)
            if self.is_local_global:
                loss = self.cri_pix(output, self.gt, self.glb_flag)
            else:
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
        if self.is_latency:
            val_output, val_latency = self.netV(self.val_img)
            loss, _ = self.cri_pix_v(val_output, self.val_gt, val_latency)
        else:
            val_output = self.netV(self.val_img)
            if self.is_local_global:
                loss = self.cri_pix_v(val_output, self.val_gt, self.val_glb_flag)  # L_val(p`)
            else:
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
        if self.is_latency:
            output, latency = self.netG(self.img)
            loss, _ = self.cri_pix(output, self.gt, latency)
        else:
            output = self.netG(self.img)
            if self.is_local_global:
                loss = self.cri_pix(output, self.gt, self.glb_flag)
            else:
                loss = self.cri_pix(output, self.gt)
        dalpha_pos = torch.autograd.grad(loss, self.netG_attr.alphas)  # dalpha { L_trn(p+) }

        # p- = p - eps*dp`
        with torch.no_grad():
            for p, d in zip(self.netG_attr.trainable_params, dp):
                if len(p) > 0 and d is not None:
                    p -= 2. * eps * d
        if self.is_latency:
            output, latency = self.netG(self.img)
            loss, _ = self.cri_pix(output, self.gt, latency)
        else:
            output = self.netG(self.img)
            if self.is_local_global:
                loss = self.cri_pix(output, self.gt, self.glb_flag)
            else:
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
