import torch.nn as nn
import torch
import pdb


class SRCNNDemosaic(nn.Module):

    def __init__(self, param_channel):
        """
        :param param_channel: number of parameter channel
        :param data_size: data size
        """
        super().__init__()
        conv1 = nn.Conv2d(4+param_channel, 64, 9, stride=1, padding=4)
        conv2 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        conv3 = nn.Conv2d(32, 12, 5, stride=1, padding=2)
        pixel_shuffle = nn.PixelShuffle(2)
        self.srcnn = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            pixel_shuffle
        )

    def forward(self, x, param_vec):
        """
        :param x: input images (N, 1, H, W), assume H and W are dividable by 2
        :param param_vec: parameter vectors of shape (batch, parameters) or None
        :return:
        """
        # input
        x = x.float()  # make sure the input is float
        H = x.size()[2]
        W = x.size()[3]

        # Bayer map -> RGGB channels
        ch_r  = x[:, :, 0::2, 0::2]
        ch_g1 = x[:, :, 0::2, 1::2]
        ch_g2 = x[:, :, 1::2, 0::2]
        ch_b  = x[:, :, 1::2, 1::2]
        x = torch.cat([ch_r, ch_g1, ch_g2, ch_b], dim=1)  # (N, 4, H/2, W/2)

        if param_vec is not None:
            feat_params = torch.unsqueeze(torch.unsqueeze(param_vec, dim=2), dim=3)  # (batch, params, 1, 1)
            feat_params = feat_params.repeat(1, 1, H//2, W//2)  # (batch, params, H/2, W/2)
            feat_in = torch.cat([x, feat_params], dim=1)
        else:
            feat_in = x

        # convolution
        output = self.srcnn(feat_in)

        return output

