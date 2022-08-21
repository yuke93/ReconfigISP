import random
import torch
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import pdb
import time
import logging
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from data.data_sampler import DistIterTrainSampler, DistIterValSampler
import math
import subprocess


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


def _init_dist_pytorch(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    if 'port' in kwargs.keys():
        kwargs.pop('port')
    rank = int(os.environ['RANK'])
    print('RANK: {}'.format(rank))
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend='nccl', port=29500, **kwargs):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        'scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(port)  # fixed port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist(args.launcher, port=args.port)
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    #### loading resume state if exists
    resume_state = None
    pass

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True

    #### create train and val dataloader
    dataset_opt = opt['datasets']['train']
    train_set = create_dataset(dataset_opt)
    total_iters = int(opt['train']['niter'])
    train_size = int(math.ceil(len(train_set) // 2 / dataset_opt['batch_size']))
    total_epochs = int(math.ceil(total_iters / train_size))
    if rank == -1:
        # info is only correct for single GPU
        logger.info('Number of train/val images: {:,d}, iters: {:,d}'.format(
            len(train_set)//2, train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
            total_epochs, total_iters))
    # sampler
    if opt['dist']:
        train_sampler = DistIterTrainSampler(train_set, world_size, rank)
        train_val_sampler = DistIterValSampler(train_set, world_size, rank)
    else:
        n_train = len(train_set)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        train_val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    # loader
    collate_fn = None
    if 'Rggb2Obj' in dataset_opt['mode']:
        collate_fn = train_set.collate_fn
    train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler, collate_fn=collate_fn)
    train_val_loader = create_dataloader(train_set, dataset_opt, opt, train_val_sampler, collate_fn=collate_fn)

    #### create model
    model_type = opt['model']
    network_type = opt['network_G']['which_model_G']
    model = create_model(opt)

    #### training
    epoch = 0
    current_step = 0
    print_freq = opt['logger']['print_freq']
    img_print_ratio = opt['logger']['image_print_ratio']
    save_checkpoint_freq = opt['logger']['save_checkpoint_freq']
    loss_type = opt['train']['pixel_criterion']
    cur_time = time.time()
    while current_step <= total_iters:
        # while loop for epoch
        epoch += 1
        if opt['dist']:
            train_sampler.set_epoch(epoch)
            train_val_sampler.set_epoch(epoch)
        for train_data, train_val_data in zip(train_loader, train_val_loader):

            # step
            current_step += 1
            if current_step > total_iters:
                break

            # data
            if 'yolo' in model_type:
                # get image and label
                img, label = train_data['noisy'], train_data['label']
                val_img, val_label = train_val_data['noisy'], train_val_data['label']
                # feed data
                data = (img, label, val_img, val_label)
            else:
                # get image
                img, gt = train_data['noisy'], train_data['gt']
                val_img, val_gt = train_val_data['noisy'], train_val_data['gt']
                # todo: meta for validation?
                if opt['meta_device']:
                    device = train_data['device']
                if 'local_global' in loss_type:
                    glb_flag = train_data['glb_flag']
                    val_glb_flag = train_val_data['glb_flag']

                # feed data
                if opt['meta_device']:
                    data = (img, gt, val_img, val_gt, device)
                elif 'local_global' in loss_type:
                    data = (img, gt, glb_flag, val_img, val_gt, val_glb_flag)
                else:
                    data = (img, gt, val_img, val_gt)
            model.feed_data(data)

            # update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            # optimization
            if model_type in ['darts', 'darts_yolo']:
                model.optimize_alphas()  # architecture step
                model.optimize_parameters()  # parameter step
            elif model_type == 'fbnet':
                model.update_temperature(current_step)  # update temperature
                model.optimize_alphas()
                model.optimize_parameters()
            elif model_type in ['isp', 'isp_yolo']:
                model.optimize_parameters()

            #### log
            if rank <= 0 and current_step % print_freq == 0:
                #-------
                print('Average time per iter: {:.6f}'.format((time.time() - cur_time) / print_freq))
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())

                # add model log
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)

                # print train_val_loss / pixel loss
                if model_type == 'darts':
                    print('train_val_loss: {}'.format(model.val_loss.item()))
                elif model_type == 'fbnet':
                    print('temperature: {}'.format(model.temperature))
                elif model_type == 'isp':
                    print('loss: {}'.format(model.l_pix.item()))
                elif model_type == 'isp_yolo':
                    loss = model.loss
                    if loss != 0:
                        loss = loss.item()
                    print('loss: {}'.format(loss))
                else:
                    pass

                # add image log, together with architecture and parameters
                if current_step % (print_freq * img_print_ratio) == 0:
                    img_np = img[0].detach().cpu().numpy()
                    # bayer -> gray if using bayer input
                    if img_np.shape[0] == 1:
                        img_np = np.concatenate([img_np] * 3, axis=0)  # 1 channel -> 3 channels

                    if 'yolo' not in model_type:
                        gt_np = gt[0].detach().cpu().numpy()
                        # bayer -> gray if bayer gt
                        if gt_np.shape[0] == 1:
                            gt_np = np.concatenate([gt_np] * 3, axis=0)

                    out_np = model.output[0].detach().cpu().numpy()
                    # bayer -> gray if bayer output
                    if out_np.shape[0] == 1:
                        out_np = np.concatenate([out_np] * 3, axis=0)
                    
                    if 'yolo' in model_type:
                        img_logs_concat = np.concatenate([img_np, out_np], axis=2)
                    else:
                        img_logs_concat = np.concatenate([img_np, out_np, gt_np], axis=2)
                    img_logs_concat = img_logs_concat[::-1, :, :]  # BGR -> RGB
                    img_logs_concat = np.transpose(img_logs_concat, [1, 2, 0])  # HxWx3
                    img_logs_concat = np.clip(img_logs_concat, 0., 1.)
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if opt['machine'] in ['st_client', 'st_sh34', 'home', 'cdc190']:
                            # use 3xHxW for above machines
                            img_logs_concat = np.transpose(img_logs_concat, [2, 0, 1])
                        tb_logger.add_image('in_out_gt_images', img_logs_concat, current_step)

                    # architecture log
                    if model_type in ['darts', 'fbnet', 'darts_yolo']:
                        alphas = model.netG_attr.alphas
                        for step_idx, alpha in enumerate(alphas):
                            a_dict = {'m_'+str(module_idx+1) : a for module_idx, a in enumerate(alpha.detach().cpu().numpy())}
                            tb_logger.add_scalars('alpha_step_'+str(step_idx+1), a_dict, current_step)

                    # pruned paths
                    if 'Prune' in network_type:
                        print('Pruned paths: {}'.format(model.netG_attr.pruned_paths))

                logger.info(message)

                cur_time = time.time()

            # save models and training states
            if rank <= 0 and current_step % save_checkpoint_freq == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)


if __name__ == '__main__':
    main()
