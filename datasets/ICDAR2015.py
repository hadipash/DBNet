from pathlib import Path
from typing import List

import numpy as np
from mindspore.dataset.vision import RandomColorAdjust, ToPIL, ToTensor

from .base_dataset import OCRDataset
from .utils import load_image, scale_pad, resize, get_bboxes


class ICDAR2015Dataset(OCRDataset):
    """IC15 DataLoader"""
    def __init__(self, path: str, augments: dict, transforms: dict, target_size: List, train: bool):
        super().__init__(augments, transforms)
        self._train = train
        # TODO: move keys to the config file
        self._keys = ['image', 'gt', 'mask', 'thresh_map', 'thresh_mask'] if train else ['image', 'polys', 'ignore']
        self._size = target_size

        path = Path(path)
        self._img_paths = [str(img_path) for img_path in path.glob(f"{'train' if train else 'test'}_images/*")]
        self._gt_paths = [path / f"{'train' if train else 'test'}_gts" / ('gt_' + Path(img_path).stem + '.txt')
                          for img_path in self._img_paths]

    def __getitem__(self, index):
        boxes = get_bboxes(self._gt_paths[index], icdar2015=True)
        data = {
            'image': load_image(self._img_paths[index]),
            'polys': boxes[0],
            'ignore': boxes[1]
        }

        # Random Augment
        if self._augments:
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
        if self._transforms:
            self._transforms(data)
        else:
            data['polys'] = np.array(data['polys'])
            data['ignore'] = np.array(data['ignore'], dtype=np.bool8)
            img, polys = scale_pad(data['image'], data['polys'], self._size)
            data['image'] = img
            data['polys'] = polys

        # Normalize
        data['image'] -= self.RGB_MEAN
        data['image'] = ToTensor()(data['image'])

        return tuple(data[k] for k in self._keys)
