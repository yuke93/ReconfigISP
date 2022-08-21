import logging

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    module_path = '/DATA/module/'
    opt_net = opt['network_G']

    # config for conditional modules
    cond_kwargs = {}
    if 'conditional_modules' in opt_net.keys():
        cond_kwargs = opt_net['conditional_modules']

    # setting parameters
    which_model = opt_net['which_model_G']
    if 'Super' in which_model:
        n_step = opt_net['n_step']
        n_modules = opt_net['n_modules']
    if 'Prune' in which_model:
        threshold = opt_net['prune_threshold']
    if 'Universal' in which_model:
        architecture = opt_net['architecture']
        if which_model == 'IspUniversal':
            individual_module_paths = opt_net['individual_module_paths']

    if which_model == 'SuperPruneFifteenDemosFourBayerTwo':
        import models.modules.super_prune_fifteen_demos_four_bayer_two as super_net
        netG = super_net.SuperPruneFifteenDemosFourBayerTwo(n_step=n_step, threshold=threshold, module_path=module_path)
    elif which_model == 'SuperPruneFifteenDemosFourBayerTwoFt':
        import models.modules.super_prune_fifteen_demos_four_bayer_two_ft as super_net
        netG = super_net.SuperPruneFifteenDemosFourBayerTwoFt(n_step=n_step, threshold=threshold, module_path=module_path)
    elif which_model == 'IspUniversal':
        import models.modules.isp_universal as isp
        netG = isp.IspUniversal(module_path=module_path,
                                indiv_module_paths=individual_module_paths,
                                architecture=architecture,
                                **cond_kwargs)
    elif which_model == 'OriginUniversal':
        import models.modules.origin_universal as isp
        netG = isp.OriginUniversal(module_path=module_path, architecture=architecture)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG
