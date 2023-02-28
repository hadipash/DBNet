from pathlib import Path
from typing import List

import numpy as np
from scipy.io import loadmat
from mindspore.dataset.vision import RandomColorAdjust, ToPIL, HWC2CHW, Normalize

from .base_dataset import OCRDataset
from .utils import load_image, scale_pad, resize


class SynthTextDataset(OCRDataset):
    def __init__(self, path: str, augments: dict, transforms: dict, target_size: List, train=True):
        super().__init__(augments, transforms)
        self._train = train
        # TODO: move keys to the config file
        self._keys = ['image', 'gt', 'mask', 'thresh_map', 'thresh_mask'] if train else ['image', 'polys', 'ignore']
        self._size = target_size
        path = Path(path)
        self._img_paths, self._boxes = self._extract_labels(path)

        self._normalize = Normalize(self._MEAN, self._STD)
        self._hwc2chw = HWC2CHW()

    @staticmethod
    def _extract_labels(path):
        mat = loadmat(path / 'gt.mat')
        images = mat['imnames'][0]
        images = [str(path / image[0]) for image in images]

        boxes = mat['wordBB'][0]
        boxes = [box.transpose().reshape(-1, 8) for box in boxes]   # to match IC15. Also some labels just have (4, 2) shape without the batch dimension
        return images, boxes


    def __getitem__(self, idx):
        data = {
            'image': load_image(self._img_paths[idx]),
            'polys': self._boxes[idx].copy(),
            'ignore': np.zeros(self._boxes[idx].shape[0], dtype=bool)
        }

        # Random Augment
        if self._augments is not None:
            self._augments(data)
            # TODO: move into augmentations
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            data['image'] = np.array(colorjitter(ToPIL()(data['image'].astype(np.uint8))), dtype=np.float32)
        else:
            data['polys'] = data['polys'].reshape((data['polys'].shape[0], -1, 2))

        img, polys = resize(data['image'], data['polys'], isTrain=self._train)
        data['image'] = img
        data['polys'] = polys

        # Post Process
        if self._transforms is not None:
            self._transforms(data)
        else:
            img, polys = scale_pad(data['image'], data['polys'], self._size)
            data['image'] = img
            data['polys'] = polys

        # Normalize
        data['image'] = self._normalize(data['image'])
        data['image'] = self._hwc2chw(data['image'])

        return tuple(data[k] for k in self._keys)
