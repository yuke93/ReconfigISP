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
from models.modules.yolo_v3_net import Darknet

logger = logging.getLogger('base')


class DartsYoloModel(BaseModel):
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

        # YOLOv3 network
        # This network needs not update, so distributed training is not used
        yolo_def = 'models/modules/yolo_v3.cfg'
        yolo_weights = '../experiments/yolo_v3/yolov3.weights'
        self.net_yolo = Darknet(yolo_def).to(self.device)
        self.net_yolo.load_darknet_weights(yolo_weights)

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
            img, label = data
        elif len(data) == 4:
            img, label, val_img, val_label = data
            self.val_img = val_img.to(self.device)
            self.val_label = val_label.to(self.device)
        else:
            raise ValueError('Invalid data format.')
        self.img = img.to(self.device)
        self.label = label.to(self.device)

    def yolo_inference(self, bgr_in, label):
        """ get (loss, yolo output) from BGR input """
        # input color space should be RGB
        output_rgb = bgr_in.clone()
        output_rgb[:, 0, :, :] = bgr_in[:, 2, :, :]
        output_rgb[:, 2, :, :] = bgr_in[:, 0, :, :]
        loss, output_yolo = self.net_yolo(output_rgb, label)
        return loss, output_yolo

    def optimize_parameters(self):
        # loss function
        self.output = self.netG(self.img)

        # yolo net
        self.loss, self.output_yolo = self.yolo_inference(self.output, self.label)

        # backward
        if self.loss == 0:
            print('[Warning] Loss is zero.')
        else:
            self.optimizer_G.zero_grad()
            self.loss.backward()
            self.optimizer_G.step()

        # set log
        self.log_dict['loss'] = 0 if self.loss == 0 else self.loss.item()

    def virtual_step(self):
        """
        Compute unrolled parameters p' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        :return valid flag (indicate loss is not 0)
        """
        # forward & calc loss
        output = self.netG(self.img)
        loss, _ = self.yolo_inference(output, self.label)
        if loss == 0:
            return False

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
        return True

    def optimize_alphas(self):
        """ Compute unrolled loss and backward its gradients """
        # zero grad
        self.optimizer_alpha.zero_grad()

        # do virtual step (calc p`)
        train_flag = self.virtual_step()
        if not train_flag:
            # loss=0
            self.optimizer_alpha.step()  # apply zero grad, may merge grads from other processes
            return

        # calc unrolled loss (on val data)
        val_output = self.netV(self.val_img)
        loss, _ = self.yolo_inference(val_output, self.val_label)  # L_val(p`)
        if loss == 0:
            self.optimizer_alpha.step()  # apply zero grad, may merge grads from other processes
            return
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
        loss, _ = self.yolo_inference(output, self.label)
        dalpha_pos = torch.autograd.grad(loss, self.netG_attr.alphas)  # dalpha { L_trn(p+) }

        # p- = p - eps*dp`
        with torch.no_grad():
            for p, d in zip(self.netG_attr.trainable_params, dp):
                if len(p) > 0 and d is not None:
                    p -= 2. * eps * d
        output = self.netG(self.img)
        loss, _ = self.yolo_inference(output, self.label)
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
