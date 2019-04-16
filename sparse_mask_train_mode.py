from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn

from torch.nn import init
from torch.nn import functional as F
from torch.utils import checkpoint as cp

from get_backbone import get_backbone


def upsample(x, h, w):
    _, _, xh, xw = x.shape

    if xh == h and xw == w:
        return x
    return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


def concat_conv(relu, down_convs, up_convs, weight, *features):
    mid = len(features) // 2
    down_feature_maps, up_feature_maps = features[:mid], features[mid:]

    x = torch.tensor(0.0).to(down_feature_maps[-1])
    _, _, h, w = down_feature_maps[-1].shape

    for idx, (down_conv, up_conv, down_feat, up_feat) in enumerate(zip(down_convs, up_convs,
                                                                       down_feature_maps, up_feature_maps)):
        x = x + weight[idx][0] * upsample(down_conv(relu(down_feat)), h, w)
        x = x + weight[idx][1] * upsample(up_conv(relu(up_feat)), h, w)
    return x


class Decoder(nn.Module):
    def __init__(self, depth, down_channels, up_channels, activation=None):
        super(Decoder, self).__init__()
        self.down_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(c, depth, kernel_size=1, bias=False),
                                                       nn.BatchNorm2d(depth, momentum=0.0003)) for c in down_channels])
        self.up_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(c, depth, kernel_size=1, bias=False),
                                                     nn.BatchNorm2d(depth, momentum=0.0003)) for c in up_channels])
        self.bn = nn.BatchNorm2d(depth, momentum=0.0003)
        self.activation = activation()

        self._weight_, self.weight = nn.Parameter(torch.ones(len(down_channels), 2)), None

    def forward(self, down_feature_maps, up_feature_maps):
        self.weight = torch.clamp(self._weight_ + torch.randn_like(self._weight_)*0.1, 0, 2)

        x = cp.checkpoint(lambda w, *features: concat_conv(
                                self.activation, self.down_convs, self.up_convs, w, *features),
                          self.weight, *(down_feature_maps+up_feature_maps))
        out = self.bn(x)

        return out


class SparseMask(nn.Module):
    def __init__(self, backbone_name, depth=64, in_channels=3, num_classes=21, activation=nn.ReLU6):
        """
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param activation:
        """
        super(SparseMask, self).__init__()

        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes
        self.depth = depth

        self.backbone = get_backbone(backbone_name, in_channels=in_channels)
        self.decoders = self._make_decoders(activation)
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
                for p in m.parameters():
                    p.requires_grad_(False)

    def _make_decoders(self, activation):
        modules = OrderedDict()
        stage_name = "Decoders"

        down_channels = list(self.backbone.output_channels())[::-1]
        up_channels = down_channels[:1] + [self.depth]*(len(down_channels)-1)
        for idx in range(len(down_channels)):
            name = stage_name + "_{}".format(idx)
            modules[name] = Decoder(self.depth, down_channels[:idx+1], up_channels[:idx+1], activation=activation)

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


def prune(model, thres=None, rm_duplicate=True, rm_subset=True):
    weights = [np.clip(np.abs(decoder._weight_.data.cpu().numpy()), 0, 2)/2 for decoder in model.decoders[::-1]]

    masks = []
    for idx, weight in enumerate(weights):
        mask = weight > thres
        if idx == 0:
            mask_history = weight > thres
        if idx > 0:
            mask *= mask_history[-idx][1]
            mask_history[:-idx] |= mask
        masks.append(mask)

    n_d = len(weights)
    total_mask = np.zeros([n_d, n_d, 2], dtype=np.bool)
    for idx, mask in enumerate(masks):
        total_mask[idx][:n_d-idx] |= mask

    # remove duplicate
    if rm_duplicate:
        for i in range(n_d-1, 0, -1):
            mask_i = total_mask[i]
            if not np.any(mask_i):
                total_mask[:, n_d-i, 1] = False
                continue
            for j in range(i-1, 0, -1):
                mask_j = total_mask[j]
                if np.all(mask_i == mask_j):
                    total_mask[j] = False
                    total_mask[:, n_d-i, 1] |= total_mask[:, n_d-j, 1]

    # remove sub set
    if rm_subset:
        for i in range(1, n_d-1):
            mask_i = total_mask[i]
            if not np.any(mask_i):
                continue
            for j in range(i+1, n_d):
                mask_j = total_mask[j]
                if np.all(mask_i*mask_j == mask_j) and np.all(total_mask[:, n_d-i, 1] *
                                                              total_mask[:, n_d-j, 1] == total_mask[:, n_d-j, 1]):
                    total_mask[:, n_d-j, 1] = False
                    total_mask[j] = False

    return total_mask[::-1]
