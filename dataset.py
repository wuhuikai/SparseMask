import os
import torch
import numpy as np

from skimage.io import imread
from skimage.transform import rescale

from torchvision import transforms
from torch.utils.data import Dataset


def scale_im(im, scale, **params):
    return rescale(im, scale, mode='constant', preserve_range=True, anti_aliasing=False, **params)


class DatasetVOC(Dataset):
    def __init__(self, im_path, gt_path, im_list, loader, training):
        self.loader = loader
        self.training = training

        self.im_path = im_path
        self.gt_path = gt_path
        self.im_list = im_list
        with open(im_list) as f:
            self.imgs = [(os.path.join(im_path, line.strip()+'.jpg'),
                          os.path.join(gt_path, line.strip()+'.png')) for line in f]

        self.crop_size = 513
        self.scale_factors = np.arange(0.5, 2.01, 0.25)

        self.imagenet_stats = {'mean': np.array([0.485, 0.456, 0.406]),
                               'std':  np.array([0.229, 0.224, 0.225])}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**self.imagenet_stats)
        ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target)
        """
        # read im & gt
        im_path, gt_path = self.imgs[index]
        im = self.loader(im_path)
        gt = self.loader(gt_path)
        # scale
        if self.training:
            scale = np.random.choice(self.scale_factors)
            im = scale_im(im, scale, multichannel=True)
            gt = scale_im(gt, scale, multichannel=False, order=0)
        # padding
        image_height, image_width, _ = im.shape
        padding_h = max(self.crop_size - image_height, 0)
        padding_w = max(self.crop_size - image_width, 0)
        l_padding_h, l_padding_w = padding_h//2, padding_w//2
        r_padding_h, r_padding_w = padding_h - l_padding_h, padding_w - l_padding_w
        im = np.stack([np.pad(im[:, :, i], ((l_padding_h, r_padding_h), (l_padding_w, r_padding_w)), 'constant',
                              constant_values=self.imagenet_stats['mean'][i]*255) for i in range(3)], axis=2)
        gt = np.pad(gt, ((l_padding_h, r_padding_h), (l_padding_w, r_padding_w)),
                    'constant', constant_values=255)
        # crop
        if self.training:
            h, w, _ = im.shape
            yl, xl = np.random.random_integers(0, h - self.crop_size), np.random.random_integers(0, w - self.crop_size)
            yr, xr = yl+self.crop_size, xl+self.crop_size
            im, gt = im[yl:yr, xl:xr], gt[yl:yr, xl:xr]
        # flip
        if self.training and np.random.rand() > 0.5:
            im, gt = np.fliplr(im), np.fliplr(gt)

        im = self.transform(im.copy().astype(np.uint8))
        gt = torch.tensor(gt.copy(), dtype=torch.long)

        return im, gt

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Image Location: {}\n'.format(self.im_path)
        fmt_str += '    GT    Location: {}\n'.format(self.gt_path)
        fmt_str += '    List  Location: {}\n'.format(self.im_list)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_loader(im_path, gt_path, im_list, batch_size, workers, training=True):
    dataset = DatasetVOC(im_path, gt_path, im_list, imread, training)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=training,
                                       num_workers=workers, pin_memory=True)
