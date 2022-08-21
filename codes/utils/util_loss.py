import torch
import torch.nn.functional as F


####################
# Loss Function
####################
def latency_loss(img_in, img_gt, latency, target_latency, w, fidelity_loss):
    """
    :param img_in: input images (N,C,H,W)
    :param img_gt: GT images (N,C,H,W)
    :param latency: latency of the current architecture
    :param target_latency: target latency
    :param w: a hyperparameter controlling the trade-off between fidelity and latency
    :param fidelity_loss: fidelity loss function
    :return: latency loss, latency term
    latency_loss = fidelity_loss(img_in, img_gt) * (latency / target_latency) ** w
    """
    # fidelity loss
    fid_loss = fidelity_loss(img_in, img_gt)
    lat_term = (latency / target_latency) ** w
    lat_loss = fid_loss * lat_term
    return lat_loss, lat_term


def local_global_loss(img_in, img_gt, glb_flag, loss_func):
    """
    :param img_in: input images (N,C,H,W)
    :param img_gt: GT images (N,C,H,W)
    :param glb_flag: global flag (N,)
    :param loss_func: loss function that takes input and GT images
    :return: loss
    """
    # local loss
    img_in_loc = img_in[glb_flag < 1]  # compare with int
    img_gt_loc = img_gt[glb_flag < 1]
    if len(img_in_loc) == 0:
        loss_loc = 0.
    else:
        # calculate the mean for each image, each channel
        in_loc_mean = img_in_loc.mean((2, 3), keepdim=True)  # (N,C,1,1)
        in_loc_mean = torch.clamp(in_loc_mean, 0, None) + 1e-6  # guarantee non-zero
        gt_loc_mean = img_gt_loc.mean((2, 3), keepdim=True)  # (N,C,1,1)
        # gain
        gain_loc = gt_loc_mean / in_loc_mean
        gain_loc = torch.clamp(gain_loc, 0.5, 2.)  # constrain the range of gain
        gain_loc = gain_loc.detach()  # no gradient
        # loss
        img_in_loc_gain = img_in_loc * gain_loc  # torch.mul
        loss_loc = loss_func(img_in_loc_gain, img_gt_loc)

    # global loss
    img_in_glb = img_in[glb_flag >= 1]
    img_gt_glb = img_gt[glb_flag >= 1]
    if len(img_in_glb) == 0:
        loss_glb = 0.
    else:
        # down sample images (scale 1/4, bilinear)
        img_in_glb_small = F.interpolate(img_in_glb, scale_factor=0.25, mode='bilinear', align_corners=False)
        img_gt_glb_small = F.interpolate(img_gt_glb, scale_factor=0.25, mode='bilinear', align_corners=False)
        # loss
        loss_glb = loss_func(img_in_glb_small, img_gt_glb_small)

    return loss_loc + loss_glb
