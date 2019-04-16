import argparse

import numpy as np

import torch
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from utils import fast_hist
from dataset import get_loader
from sparse_mask_eval_mode import SparseMask

import warnings
warnings.simplefilter("ignore")


def evaluate(args):
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
    mask = np.load(args.mask_path)
    model = SparseMask(mask, backbone_name=args.backbone_name, depth=args.depth, in_channels=3, num_classes=args.n_class)
    # dataset
    eval_loader = get_loader(args.im_path, args.gt_path, args.eval_list, 1, 1, training=False)
    # to device
    if args.gpu >= 0 is not None:
        model = torch.nn.DataParallel(model, [args.gpu])
    model.to(device=device, dtype=dtype)
    # load weight
    checkpoint = torch.load(args.pretrained_model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    model.eval()
    with torch.no_grad():
        hist = np.zeros((args.n_class, args.n_class))
        for batch_idx, (data, target) in enumerate(tqdm(eval_loader)):
            data = data.to(device=device, dtype=dtype)
            output = model(data)

            _, h, w = target.shape
            output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

            output, target = output.data.cpu().numpy(), target.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            hist += fast_hist(target.flatten(), output.flatten(), args.n_class)

    m_iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print(np.sum(m_iou) / len(m_iou))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval SparseMask')
    parser.add_argument('--depth', default=64, type=int)
    parser.add_argument('--mask_path', default=None)
    parser.add_argument('--backbone_name', default='mobilenet_v2')
    # Dataset
    parser.add_argument('--im_path', default='VOC12/data/img')
    parser.add_argument('--gt_path', default='VOC12/data/gt')
    parser.add_argument('--n_class', default=21, type=int)
    parser.add_argument('--eval_list', default='VOC12/data/val.txt')
    # Device
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--type', default='float32')
    # Checkpoints
    parser.add_argument('--pretrained_model', required=True)

    args = parser.parse_args()
    # train
    evaluate(args)
