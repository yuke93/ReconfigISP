import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'darts':
        from .darts_model import DartsModel as M
    elif model == 'darts_yolo':
        from .darts_yolo_model import DartsYoloModel as M
    elif model == 'darts_ft':
        from .darts_ft_model import DartsFtModel as M
    elif model == 'isp':
        from .isp_model import IspModel as M
    elif model == 'isp_yolo':
        from .isp_yolo_model import IspYoloModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(model))
    return m
