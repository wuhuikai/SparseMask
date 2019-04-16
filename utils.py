import numpy as np

from PIL import Image
from tqdm import tqdm


class TqdmStream(object):
    @classmethod
    def write(cls, msg):
        tqdm.write(msg, end='')

    @classmethod
    def flush(cls):
        pass


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def decode_labels(mask, num_classes=21):
    # colour map
    label_colours = [(0, 0, 0)
                     # 0=background
                     , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                     # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                     , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                     # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                     , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                     # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                     , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
                     # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    h, w = mask.shape

    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for j_, j in enumerate(mask[:, :]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = label_colours[k]
            else:
                pixels[k_, j_] = (255, 255, 255)
    output = np.array(img).transpose(2, 0, 1)

    return output
