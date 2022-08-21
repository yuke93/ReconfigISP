import torch.nn as nn
import torch
import pdb


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, outchannel, 3, 1, 1),  # with bias, no BN
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1),  # with bias, no BN
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return out


class Path14lBayer(nn.Module):

    def __init__(self, param_channel):
        """
        :param param_channel: number of parameter channel
        :param data_size: data size
        """
        super().__init__()
        # do not use global info
        # 14-layer Path-Restore with residual learning, for Bayer denoising
        # input 4 channels, RGGB, output 1 channel in Bayer domain

        # first convolution
        conv_first = nn.Conv2d(4+param_channel, 64, 3, stride=1, padding=1)

        # middle layers
        middle_layers = []
        for idx in range(6):  # 6 residual blocks, 12 layers
            middle_layers.append(ResidualBlock(inchannel=64, outchannel=64))
        conv_middle = nn.Sequential(*middle_layers)

        # last convolution (include relu and pixel shuffle)
        relu_last = nn.ReLU(inplace=True)
        conv_last = nn.Conv2d(64, 4, 3, stride=1, padding=1)
        pixel_shuffle = nn.PixelShuffle(2)

        # 14-layer Path-Restore
        self.path_restore_14l = nn.Sequential(
            conv_first,
            conv_middle,
            relu_last,
            conv_last,
            pixel_shuffle,
        )

    def forward(self, x, param_vec):
        """
        :param x: input images (N, C, H, W), assume H and W are dividable by 2
        :param param_vec: parameter vectors of shape (batch, parameters)
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
            feat_params = param_vec  # (batch, params)
            feat_params = torch.unsqueeze(torch.unsqueeze(feat_params, dim=2), dim=3)  # (batch, params, 1, 1)
            feat_params = feat_params.repeat(1, 1, H//2, W//2)  # (batch, params, H/2, W/2)
            feat_in = torch.cat([x, feat_params], dim=1)
        else:
            feat_in = x

        # convolution
        output = self.path_restore_14l(feat_in)  # (N, 3, H, W)

        return output

