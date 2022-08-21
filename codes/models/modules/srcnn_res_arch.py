import torch.nn as nn
import torch
import pdb


class SRCNNRes(nn.Module):

    def __init__(self, param_channel):
        """
        :param param_channel: number of parameter channel
        :param data_size: data size
        """
        super().__init__()
        # use global info min, mean, max as the input (each has 3 channels)
        conv1 = nn.Conv2d(3+9+param_channel, 64, 9, stride=1, padding=4)
        conv2 = nn.Conv2d(64, 32, 5, stride=1, padding=2)
        conv3 = nn.Conv2d(32, 3, 5, stride=1, padding=2)
        self.srcnn = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
        )

    def forward(self, x, param_vec):
        """
        :param x: input images (N, C, H, W)
        :param param_vec: parameter vectors of shape (batch, parameters)
        :return:
        """
        # input
        x = x.float()  # make sure the input is float
        H = x.size()[2]
        W = x.size()[3]
        feat_min, _ = torch.min(x, dim=3)  # pytorch 1.0.1 interface
        feat_min, _ = torch.min(feat_min, dim=2)
        feat_mean = torch.mean(torch.mean(x, dim=3), dim=2)
        feat_max, _ = torch.max(x, dim=3)
        feat_max, _ = torch.max(feat_max, dim=2)
        try:
            feat_params = torch.cat([feat_min, feat_mean, feat_max, param_vec], dim=1)  # (batch, 3 + params)
        except:
            raise ValueError(feat_min.size(), param_vec.size())
        feat_params = torch.unsqueeze(torch.unsqueeze(feat_params, dim=2), dim=3)  # (batch, 3 + params, 1, 1)
        feat_params = feat_params.repeat(1, 1, H, W)  # (batch, 3 + params, H, W)
        feat_in = torch.cat([x, feat_params], dim=1)

        # convolution
        residual = self.srcnn(feat_in)
        output = x + residual

        return output

