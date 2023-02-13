from  typing import Callable
from abc import ABC, abstractmethod

import numpy as np
# from mindcv.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .process import MakeBorderMap, MakeSegDetectionData
from .augment.augments import RandomCrop, RandomFlip, RandomScale, RandomRotate


_AUGMENTATIONS = {
    'RandomCrop': RandomCrop,
    'RandomFlip': RandomFlip,
    'RandomScale': RandomScale,
    'RandomRotate': RandomRotate
}


_TRANSFORMS = {
    'MakeBorderMap': MakeBorderMap,
    'MakeSegDetectionData': MakeSegDetectionData
}


class OCRDataset(ABC):
    def __init__(self, augments: dict, transforms: dict):
        self._img_paths = []
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

        self._augments = self._process(augments, _AUGMENTATIONS) if augments else None
        self._transforms = self._process(transforms, _TRANSFORMS) if augments else None

    @staticmethod
    def _process(ops: dict, func: dict) -> Callable:
        initialized_ops = []
        for op, params in ops.items():
            initialized_ops.append(func[op](**params))

        def convert(data):
            for iop in initialized_ops:
                iop(data)
        return convert

    @abstractmethod
    def __getitem__(self, idx):
        ...

    def __len__(self):
        return len(self._img_paths)
