import os
import glob
import shutil
import random
import logging
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

from datetime import datetime
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from dataset import get_loader
from utils import TqdmStream, fast_hist, decode_labels

import warnings
warnings.simplefilter("ignore")


def loss_calc(criterion, output, target):
    _, h, w = target.shape
    output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
    return criterion(output, target)


def lr_poly(iter, max_iter, power):
    return (1 - float(iter) / max_iter) ** power


def get_backbone_params(model):
    for key, value in model.named_parameters():
        if 'backbone' in key:
            yield value


def get_decoder_params(model):
    for key, value in model.named_parameters():
        if 'backbone' not in key:
            yield value


def evaluate(model, loader, n_class, device, dtype, iter_idx, writer):
    hist = np.zeros((n_class, n_class))

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device=device, dtype=dtype)

        with torch.no_grad():
            output = model(data)
            _, h, w = target.shape
            output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

        output, target = output.data.cpu().numpy(), target.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        hist += fast_hist(target.flatten(), output.flatten(), n_class)

        if batch_idx == 0:
            writer.add_image('val/input', vutils.make_grid(data, normalize=True, scale_each=True, padding=0), iter_idx)
            writer.add_image('val/output', decode_labels(output[0]), iter_idx)
            writer.add_image('val/gt', decode_labels(target[0]), iter_idx)

    m_iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return np.sum(m_iou) / len(m_iou)


def get_model(args):
    if args.search:
        from sparse_mask_train_mode import SparseMask
        from train_sparse_mask_search_mode import run
        model = SparseMask(backbone_name=args.backbone_name, depth=args.depth, in_channels=3, num_classes=args.n_class)
        return model, run
    else:
        from sparse_mask_eval_mode import SparseMask
        from train_sparse_mask_prune_mode import run
        mask = np.load(args.mask_path)
        model = SparseMask(mask, backbone_name=args.backbone_name, depth=args.depth, in_channels=3, num_classes=args.n_class)
        return model, run


def train(args):
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
    # model
    model, run = get_model(args)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info(model)
    logging.info('number of parameters: {}'.format(num_parameters))
    # dataset
    train_loader = get_loader(args.im_path, args.gt_path, args.training_list, args.batch_size, args.n_worker)
    eval_loader = get_loader(args.im_path, args.gt_path, args.eval_list, 1, args.n_worker, training=False)
    # loss
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # use a Classification Cross-Entropy loss
    # to device
    if args.gpu >= 0 is not None:
        model = torch.nn.DataParallel(model, [args.gpu])
    model.to(device=device, dtype=dtype)
    criterion.to(device=device, dtype=dtype)
    # load weight
    logging.info("=> loading checkpoint '{}'".format(args.pretrained_model))
    checkpoint = torch.load(args.pretrained_model, map_location=device)
    model.module.backbone.load_state_dict(checkpoint['state_dict'], strict=False)
    logging.info("=> loaded checkpoint '{}'".format(args.pretrained_model))
    # load weight
    if args.ft_model is not None:
        logging.info("=> loading ft model '{}'".format(args.ft_model))
        checkpoint = torch.load(args.ft_model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logging.info("=> loaded ft model '{}'".format(args.ft_model))
    # optimizer
    max_iter = args.max_epoch*len(train_loader)
    optimizer = optim.SGD([{'params': get_backbone_params(model), 'lr': args.lr},
                           {'params': get_decoder_params(model), 'lr': args.last_layer_lr_mult*args.lr}],
                          lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda iter: lr_poly(iter, max_iter, args.gamma),
                                               lambda iter: lr_poly(iter, max_iter, args.gamma)])

    model.train()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    for epoch in trange(args.max_epoch):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            iter_idx = epoch * len(train_loader) + batch_idx

            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
            run(model, criterion, optimizer, data, target, scheduler, iter_idx, args)

        iter_idx = (epoch+1) * len(train_loader)
        torch.save({'iter': iter_idx, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(args.exp_path, 'checkpoint_{}.pth.tar'.format(iter_idx)))
        model.eval()
        m_iou = evaluate(model, eval_loader, args.n_class, device, dtype, iter_idx, args.writer)
        model.train()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        logging.info('Train Epoch: {} [{}/{} ({:.0f}%)], mIOU: {:.6f}'
                     ', 1x_lr: {}, 10x_lr: {}'.format(epoch, batch_idx, len(train_loader),
                                                      100. * batch_idx / len(train_loader), m_iou,
                                                      optimizer.param_groups[0]['lr'],
                                                      optimizer.param_groups[1]['lr']))
        args.writer.add_scalar('val/m_iou', m_iou, iter_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SparseMask')
    parser.add_argument('--search', default=False, action='store_true')
    parser.add_argument('--depth', default=64, type=int)
    parser.add_argument('--mask_path', default=None)
    parser.add_argument('--backbone_name', default='mobilenet_v2')
    # Dataset
    parser.add_argument('--im_path', default='VOC12/data/img')
    parser.add_argument('--gt_path', default='VOC12/data/gt')
    parser.add_argument('--n_class', default=21, type=int)
    parser.add_argument('--training_list', default='VOC12/data/train_aug.txt')
    parser.add_argument('--eval_list', default='VOC12/data/val.txt')
    # Device
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--n_worker', default=4, type=int)
    parser.add_argument('--type', default='float32')
    # Training
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--last_layer_lr_mult', default=10.0, type=float)
    parser.add_argument('--decay', default=0.00004, type=float)
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--freeze_bn', default=False, action='store_true')
    # Sparse
    parser.add_argument('--sparse', default=1e-2, type=float)
    # Checkpoints
    parser.add_argument('--exp_path', default='SparseMaskVOC')
    parser.add_argument('--pretrained_model', default='models/mobilenet_v2_1.0_224/model_best.pth.tar')
    parser.add_argument('--ft_model', default=None)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_interval', default=100, type=int)

    args = parser.parse_args()
    # seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # log dir
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.exp_path = os.path.join(args.exp_path, time_stamp)
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
    # logger
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join(args.exp_path, 'log.txt')),
                            logging.StreamHandler(TqdmStream)
                        ],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # args
    logging.info(args)
    # copy code
    source_dir = os.path.join(args.exp_path, 'source')
    os.makedirs(source_dir)
    for filename in glob.glob(os.path.join('.', '*.py')):
        shutil.copy(filename, source_dir)
    # tensorboard
    args.writer = SummaryWriter(os.path.join(args.exp_path, 'tensorboard_log'))
    # train
    train(args)
    # close tensorboard
    args.writer.export_scalars_to_json(os.path.join(args.exp_path, 'tensorboard.json'))
    args.writer.close()
