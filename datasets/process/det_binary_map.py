import cv2
import numpy as np
from shapely.geometry import Polygon

from .utils import expand_poly


class ShrunkBinaryMap:
    """
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    """
    def __init__(self, min_text_size=8, shrink_ratio=0.4, train=True):
        self._min_text_size = min_text_size
        self._shrink_ratio = shrink_ratio
        self._train = train
        self._dist_coef = 1 - self._shrink_ratio ** 2

    def __call__(self, data):
        if self._train:
            self._validate_polys(data)

        gt = np.zeros(data['image'].shape[:2], dtype=np.float32)
        mask = np.ones(data['image'].shape[:2], dtype=np.float32)
        for i in range(len(data['polys'])):
            min_side = min(np.max(data['polys'][i], axis=0) - np.min(data['polys'][i], axis=0))

            if data['ignore'][i] or min_side < self._min_text_size:
                cv2.fillPoly(mask, [data['polys'][i].astype(np.int32)], 0)
                data['ignore'][i] = True
            else:
                poly = Polygon(data['polys'][i])
                shrunk = expand_poly(data['polys'][i], distance=-self._dist_coef * poly.area / poly.length)

                if shrunk:
                    cv2.fillPoly(gt, [np.array(shrunk[0], dtype=np.int32)], 1)
                else:
                    cv2.fillPoly(mask, [data['polys'][i].astype(np.int32)], 0)
                    data['ignore'][i] = True

        data['gt'] = np.expand_dims(gt, axis=0)
        data['mask'] = mask

    @staticmethod
    def _validate_polys(data):
        data['polys'] = np.clip(data['polys'], 0, np.array(data['image'].shape[1::-1]) - 1)  # shape reverse order: w, h

        for i in range(len(data['polys'])):
            poly = Polygon(data['polys'][i])
            if poly.area < 1:
                data['ignore'][i] = True
            if not poly.exterior.is_ccw:
                data['polys'][i] = data['polys'][i][::-1]
