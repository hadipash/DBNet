# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# This file refers to the project https://github.com/MhLiao/DB.git

"""DBNet Dataset pre process functions."""

import cv2
import numpy as np
import imgaug.augmenters as aug_img
import imgaug


def solve_polys(polys):
    """Group poly together."""
    max_points = 0
    for poly in polys:
        if len(poly) // 2 > max_points:
            max_points = len(poly) // 2
    new_polys = []
    for poly in polys:
        new_poly = []
        if len(poly) // 2 < max_points:
            new_poly.extend(poly)
            for _ in range(len(poly) // 2, max_points):
                new_poly.extend([poly[0], poly[1]])
        else:
            new_poly = poly
        new_polys.append(new_poly)
    return np.array(new_polys), max_points


class RandomCrop:
    """Random crop class, include many crop relevant functions."""
    def __init__(self, max_tries=10, min_crop_side_ratio=0.1, crop_size=(640, 640)):
        self.size = crop_size
        self.min_crop_side_ratio = min_crop_side_ratio
        self.max_tries = max_tries

    def __call__(self, data):
        # Eliminate dontcare polys.
        all_care_polys = [data['polys'][i] for i in range(len(data['ignore'])) if not data['ignore'][i]]
        # Crop a rectangle randomly.
        crop_x, crop_y, crop_w, crop_h = self.crop_area(data['image'], all_care_polys)
        # Rescale the cropped rectangle to crop_size.
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        # Pad the rest of crop_size with 0.
        padimg = np.zeros((self.size[1], self.size[0], data['image'].shape[2]), data['image'].dtype)
        start_y, start_x = np.random.randint(0, self.size[1] - h + 1), np.random.randint(0, self.size[0] - w + 1)
        padimg[start_y: start_y + h, start_x: start_x + w] = cv2.resize(
            data['image'][crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
        data['image'] = padimg

        new_polys = []
        new_dontcare = []
        for i in range(len(data['polys'])):
            # Rescale all original polys.
            poly = data['polys'][i]
            poly = ((poly - (crop_x, crop_y)) * scale)
            # Filter out the polys in the cropped rectangle.
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                new_polys.append(poly + (start_x, start_y))
                new_dontcare.append(data['ignore'][i])

        data['polys'] = new_polys
        data['ignore'] = new_dontcare

    def is_poly_in_rect(self, poly, x, y, w, h):
        '''
        Whether the poly is inside a rectangle.
        '''
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        '''
        Whether the poly isn't inside a rectangle.
        '''
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        '''
        Splitting out the continuous area in the axis.
        '''
        regions = []
        min_axis = 0
        for i in range(1, len(axis)):
            # If continuous
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        '''
        Randomly select two values in a single region.
        '''
        xx = np.random.choice(axis, size=2)
        xmin = np.clip(np.min(xx), 0, max_size - 1)
        xmax = np.clip(np.max(xx), 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        '''
        Two regions are randomly selected from regions and then one value is taken from each.
        Return the two values taken.
        '''
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)

        xmin = np.clip(min(selected_values), 0, max_size - 1)
        xmax = np.clip(max(selected_values), 0, max_size - 1)
        return xmin, xmax

    def crop_area(self, img, polys):
        '''
        Randomly select a rectangle containing polys from the img.
        Return the start point and side lengths of the selected rectangle.
        '''
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)

        for points in polys:
            # Convert points from float to int.
            points = np.round(points, decimals=0).astype(np.int32)
            # interval of x
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            # interval of y
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # Get the idx that include text.
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if (not h_axis.any()) or (not w_axis.any()):
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for _ in range(self.max_tries):
            # Randomly select two contained idx in the axis to form a new rectangle.
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)
            # If too small, reselect.
            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                continue
            # If there is a poly inside the rectangle, successful.
            num_poly_in_rect = 0
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        # If the num of attempts exceeds 'max_tries', return the whole img.
        return 0, 0, w, h


def _augment_poly(aug, img_shape, poly):
    keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
    keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints, shape=img_shape[:2])])
    keypoints = keypoints[0].keypoints
    poly = [(p.x, p.y) for p in keypoints]
    return np.array(poly)


class RandomRotate:
    def __init__(self, random_angle):
        self._random_range = random_angle

    def __call__(self, data):
        angle = np.random.randint(self._random_range[0], self._random_range[1])
        aug_bin = aug_img.Sequential([aug_img.Affine(rotate=angle)])
        data['image'] = aug_bin.augment_image(data['image'])

        new_polys = []
        for poly in data['polys']:
            poly = _augment_poly(aug_bin, data['image'].shape, poly)
            poly = np.maximum(poly, 0)
            new_polys.append(poly)
        data['polys'] = new_polys


class RandomScale:
    def __init__(self, short_side):
        self._short_side = short_side

    def __call__(self, data):
        polys, max_points = solve_polys(data['polys'])
        h, w = data['image'].shape[0:2]

        # polys -> polys' scale w.r.t original.
        polys_scale = []
        for poly in polys:
            poly = np.asarray(poly)
            poly = poly / ([w * 1.0, h * 1.0] * max_points)
            polys_scale.append(poly)
        polys_scale = np.array(polys_scale)

        # Resize to 1280 pixs max-length.
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            data['image'] = cv2.resize(data['image'], dsize=None, fx=scale, fy=scale)
        h, w = data['image'].shape[0:2]

        # Get scale randomly.
        random_scale = np.array([0.5, 1.0, 2.0, 3.0])
        scale = np.random.choice(random_scale)
        # If less than short_side, scale will be clipped to min_scale.
        if min(h, w) * scale <= self._short_side:
            scale = (self._short_side + 10) * 1.0 / min(h, w)
        # Rescale img.
        data['image'] = cv2.resize(data['image'], dsize=None, fx=scale, fy=scale)
        # Rescale polys: (N, 8) -> (N, 4, 2)
        data['polys'] = (polys_scale * ([data['image'].shape[1], data['image'].shape[0]] * max_points))\
            .reshape((polys.shape[0], polys.shape[1] // 2, 2))


class RandomFlip:
    def __call__(self, data):
        if np.random.rand(1)[0] > 0.5:
            aug_bin = aug_img.Sequential([aug_img.Fliplr((1))])
            data['image'] = aug_bin.augment_image(data['image'])
            new_polys = []
            for poly in data['polys']:
                poly = _augment_poly(aug_bin, data['image'].shape, poly)
                poly = np.maximum(poly, 0)
                new_polys.append(poly)
            data['polys'] = new_polys
