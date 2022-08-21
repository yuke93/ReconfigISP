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


# Note this model takes BGR rather than RGB as input
class Path14lBgr(nn.Module):

    def __init__(self, param_channel):
        """
        :param param_channel: number of parameter channel
        :param data_size: data size
        """
        super().__init__()
        # do not use global info
        # 14-layer Path-Restore. RGB domain
        # input 3 channels, output 3 channels

        # first convolution
        conv_first = nn.Conv2d(3+param_channel, 64, 3, stride=1, padding=1)

        # middle layers
        middle_layers = []
        for idx in range(6):  # 6 residual blocks, 12 layers
            middle_layers.append(ResidualBlock(inchannel=64, outchannel=64))
        conv_middle = nn.Sequential(*middle_layers)

        # last convolution (include relu and pixel shuffle)
        relu_last = nn.ReLU(inplace=True)
        conv_last = nn.Conv2d(64, 3, 3, stride=1, padding=1)

        # 14-layer Path-Restore
        self.path_restore_14l = nn.Sequential(
            conv_first,
            conv_middle,
            relu_last,
            conv_last
        )

    def forward(self, x, param_vec):
        """
        :param x: input images (N, C, H, W)
        :param param_vec: parameter vectors of shape (batch, parameters)
        :return:
        """
        # BGR -> RGB
        x = x[:, [2, 1, 0], :, :]

        # input
        x = x.float()  # make sure the input is float
        H = x.size()[2]
        W = x.size()[3]

        if param_vec is not None:
            feat_params = param_vec  # (batch, params)
            feat_params = torch.unsqueeze(torch.unsqueeze(feat_params, dim=2), dim=3)  # (batch, params, 1, 1)
            feat_params = feat_params.repeat((1, 1, H, W))  # (batch, params, H, W)
            feat_in = torch.cat((x, feat_params), dim=1)
        else:
            feat_in = x

        # convolution
        output = self.path_restore_14l(feat_in)  # (N, 3, H, W)

        # RGB -> BGR
        output = output[:, [2, 1, 0], :, :]

        return output
