import glob
import os

import cv2
import numpy as np
from mindspore.dataset.vision import RandomColorAdjust, ToPIL, ToTensor

from datasets.utils import load_image, resize


class TotalTextDataset:
    def __init__(self, config, isTrain=True):
        if isTrain:
            img_paths = glob.glob(os.path.join(config['train']['img_dir'],
                                               '*' + config['train']['img_format']))
        else:
            img_paths = glob.glob(os.path.join(config['eval']['img_dir'],
                                               '*' + config['eval']['img_format']))
        self.img_paths = img_paths
        if self.isTrain:
            img_dir = config['train']['gt_dir']
            gt_paths = [os.path.join(img_dir, 'poly_gt_' + img_path.split('/')[-1].split('.')[0] + '.mat')
                        for img_path in self.img_paths]

        else:
            img_dir = config['eval']['gt_dir']
            gt_paths = [os.path.join(img_dir, 'poly_gt_' + img_path.split('/')[-1].split('.')[0] + '.mat')
                        for img_path in self.img_paths]

        self.gt_paths = gt_paths

    def get_bboxes(self, gt_path):
        """
        Process a mat, that is, process the polygons of an image
        :param anno_file: mat file path
        """
        from scipy.io import loadmat
        gt_dict = loadmat(gt_path)
        arr = gt_dict['polygt']
        polys = []
        dontcare = []
        for line in range(0, arr.shape[0]):
            x_arr = arr[line][1]
            y_arr = arr[line][3]
            content = arr[line][4]
            if content.shape[0] == 0:
                content_str = '#'
            else:
                content_str = content.item()
            poly = []
            for i in range(0, x_arr.shape[1]):
                poly.append(x_arr[0][i])
                poly.append(y_arr[0][i])
            polys.append(poly)
            if "#" in content_str:
                dontcare.append(True)
            else:
                dontcare.append(False)

        return polys, np.array(dontcare)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        img = load_image(img_path)
        original = resize(img)
        polys, dontcare = self.get_bboxes(gt_path)

        if self.isTrain and self.config['train']['is_transform']:
            img, polys = self.ra.random_scale(img, polys, self.config['dataset']['short_side'])
            img, polys = self.ra.random_rotate(img, polys, self.config['dataset']['random_angle'])
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop(img, polys, dontcare)
        else:
            polys = self.polys_convert(polys)

        img, polys = resize(img, polys, isTrain=self.isTrain)
        if self.isTrain:
            img, gt, gt_mask = self.ms(img, polys, dontcare)
            img, thresh_map, thresh_mask = self.mb(img, polys, dontcare)
        else:
            dontcare = np.array(dontcare, dtype=np.bool8)

        # Show Images
        if self.config['dataset']['is_show']:
            cv2.imwrite('./images/img.jpg', img)
            cv2.imwrite('./images/gt.jpg', gt[0] * 255)
            cv2.imwrite('./images/gt_mask.jpg', gt_mask * 255)
            cv2.imwrite('./images/thresh_map.jpg', thresh_map * 255)
            cv2.imwrite('./images/thresh_mask.jpg', thresh_mask * 255)

        # Random Colorize
        if self.isTrain and self.config['train']['is_transform']:
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(ToPIL()(img.astype(np.uint8)))

        # Normalize
        img -= self.RGB_MEAN
        img = ToTensor()(img)

        if self.isTrain:
            return img, gt, gt_mask, thresh_map, thresh_mask
        return original, img, polys, dontcare
