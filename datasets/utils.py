import math

import cv2
import numpy as np


def load_image(img_path, as_float=True):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if as_float:
        img = img.astype('float32')
    return img


def scale_pad(img, polys, eval_size):
    """scale image and polys with short side, then pad to eval_size."""
    h, w, c = img.shape
    s_h = eval_size[0] / h
    s_w = eval_size[1] / w
    scale = min(s_h, s_w)
    new_h = int(scale * h)
    new_w = int(scale * w)
    img = cv2.resize(img, (new_w, new_h))
    padimg = np.zeros((eval_size[0], eval_size[1], c), img.dtype)
    padimg[:new_h, :new_w, :] = img
    polys = polys * scale
    return padimg, polys


def get_bboxes(gt_path, icdar2015=False):
    """Get polys and it's `dontcare` flag by gt_path."""
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    polys = []
    dontcare = []
    for line in lines:
        line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        if "#" in gt[-1]:
            dontcare.append(True)
        else:
            dontcare.append(False)
        if icdar2015:
            box = [int(gt[i]) for i in range(8)]
        else:
            box = [int(gt[i]) for i in range(len(gt) - 1)]
        polys.append(box)
    return np.array(polys), np.array(dontcare)


def resize(img, polys=None, denominator=32, isTrain=True):
    """Resize image and its polys."""
    w_scale = math.ceil(img.shape[1] / denominator) * denominator / img.shape[1]
    h_scale = math.ceil(img.shape[0] / denominator) * denominator / img.shape[0]
    img = cv2.resize(img, dsize=None, fx=w_scale, fy=h_scale)
    if polys is None:
        return img
    if isTrain:
        new_polys = []
        for poly in polys:
            poly[:, 0] = poly[:, 0] * w_scale
            poly[:, 1] = poly[:, 1] * h_scale
            new_polys.append(poly)
        polys = new_polys
    else:
        polys[:, :, 0] = polys[:, :, 0] * w_scale
        polys[:, :, 1] = polys[:, :, 1] * h_scale
    return img, polys
