import os
import argparse

import numpy as np

import torch
import torch.backends.cudnn as cudnn

from sparse_mask_train_mode import SparseMask, prune


def prune_model(args):
    # device
    device = torch.device('cuda:{}'.format(args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        cudnn.benchmark = True
    # dtype
    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')
    # prune
    model = SparseMask(backbone_name=args.backbone_name, depth=args.depth, in_channels=3, num_classes=args.n_class)
    if args.gpu >= 0:
        model = torch.nn.DataParallel(model, [args.gpu])
    model.to(device=device, dtype=dtype)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    mask = prune(model.module if args.gpu >= 0 else model, args.thres)
    np.save(os.path.join(os.path.dirname(args.checkpoint), 'mask_thres_{}'.format(args.thres)), mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prune SparseMask')
    # Dataset
    parser.add_argument('--n_class', default=21, type=int)
    # Model
    parser.add_argument('--backbone_name', default='mobilenet_v2')
    parser.add_argument('--depth', default=64, type=int)
    # Device
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--type', default='float32')
    # Prune
    parser.add_argument('--thres', type=float, default=1e-3)
    # Checkpoints
    parser.add_argument('--checkpoint', required=True)

    args = parser.parse_args()
    # prune
    prune_model(args)
