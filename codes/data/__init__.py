'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None, collate_fn=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = dist.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True if sampler is None else False
        collate_fn = default_collate if collate_fn is None else collate_fn
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False, collate_fn=collate_fn)
    elif phase == 'test':
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,  # might set worker=1
                                           pin_memory=True)
    else:
        raise ValueError('Unknown phase: {}'.format(phase))


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'SID_Sony_Ratio_RGGB2BGR':
        from data.sid_sony_ratio_rggb2bgr_dataset import SidSonyRatioRggb2BgrDataset as D
    elif mode == 'SID_Sony_Ratio_Test_RGGB2BGR':
        from data.sid_sony_ratio_test_rggb2bgr_dataset import SidSonyRatioTestRggb2BgrDataset as D
    elif mode == 'S7ISP_RGGB2BGR':
        from data.s7isp_rggb2bgr_dataset import S7ispRggb2BgrDataset as D
    elif mode == 'S7ISP_RGGB2BGR_Test':
        from data.s7isp_rggb2bgr_test_dataset import S7ispRggb2BgrTestDataset as D
    elif mode == 'OnePlus_Rggb2Obj':
        from data.oneplus_rggb2obj_dataset import OnePlusRggb2ObjDataset as D
    elif mode == 'OnePlus_Rggb2Obj_Test':
        from data.oneplus_rggb2obj_test_dataset import OnePlusRggb2ObjTestDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset {:s} is created.'.format(mode))
    return dataset
