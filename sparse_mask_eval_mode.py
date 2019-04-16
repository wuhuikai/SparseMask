from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn

from torch.nn import init
from torch.nn import functional as F

from get_backbone import get_backbone


def upsample(x, h, w):
    _, _, xh, xw = x.shape

    if xh == h and xw == w:
        return x
    return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


def concat_conv(relu, down_convs, up_convs, down_feature_maps, up_feature_maps):
    x = torch.tensor(0.0).to(down_feature_maps[-1])
    _, _, h, w = down_feature_maps[-1].shape

    for idx, (down_conv, up_conv, down_feat, up_feat) in enumerate(zip(down_convs, up_convs,
                                                                       down_feature_maps, up_feature_maps)):
        if down_conv:
            x = x + upsample(down_conv(relu(down_feat)), h, w)
        if up_conv:
            x = x + upsample(up_conv(relu(up_feat)), h, w)
    return x


class Decoder(nn.Module):
    def __init__(self, depth, down_channels, up_channels, mask, activation=None):
        super(Decoder, self).__init__()
        self.none = not np.any(mask)

        if not self.none:
            self.down_convs = nn.ModuleList([nn.Conv2d(c, depth, kernel_size=1, bias=False) if m else None
                                             for c, m in zip(down_channels, mask[:, 0])])
            self.up_convs = nn.ModuleList([nn.Conv2d(c, depth, kernel_size=1, bias=False) if m else None
                                           for c, m in zip(up_channels, mask[:, 1])])
            self.bn = nn.BatchNorm2d(depth, momentum=0.0003)
            self.activation = activation()

    def forward(self, down_feature_maps, up_feature_maps):
        if self.none:
            return torch.tensor(0.0).to(down_feature_maps[-1])

        x = concat_conv(self.activation, self.down_convs, self.up_convs, down_feature_maps, up_feature_maps)
        out = self.bn(x)

        return out


class SparseMask(nn.Module):
    def __init__(self, mask, backbone_name, depth=64, in_channels=3, num_classes=21, activation=nn.ReLU6):
        """
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param num_classes: number of classes to predict.
        :param activation:
        """
        super(SparseMask, self).__init__()

        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes
        self.depth = depth

        self.backbone = get_backbone(backbone_name, in_channels=in_channels)
        self.decoders = self._make_decoders(mask, activation)
        self.fc_conv = nn.Conv2d(depth, self.num_classes, kernel_size=1, bias=True)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _make_decoders(self, mask, activation):
        modules = OrderedDict()
        stage_name = "Decoders"

        down_channels = list(self.backbone.output_channels())[::-1]
        up_channels = down_channels[:1] + [self.depth]*(len(down_channels)-1)
        for idx in range(len(down_channels)):
            name = stage_name + "_{}".format(idx)
            modules[name] = Decoder(self.depth, down_channels[:idx+1], up_channels[:idx+1], mask[idx, :idx+1], activation=activation)

        return nn.Sequential(modules)

    def forward(self, x):
        down_feature_maps = self.backbone.forward(x)
        # DECODE
        down_feature_maps = down_feature_maps[::-1]
        up_feature_maps = [F.adaptive_avg_pool2d(down_feature_maps[0], 1)]
        for idx in range(len(self.decoders)):
            x = self.decoders[idx](down_feature_maps[:idx+1], up_feature_maps)
            if idx < len(self.decoders) - 1:
                up_feature_maps.append(x)

        x = self.activation(x)
        x = self.fc_conv(x)

        return x
