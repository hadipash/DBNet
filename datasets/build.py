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

"""DBNet Dataset DataLoader"""
from typing import Tuple
from pprint import pformat

from mindspore import dataset as ds

from .ICDAR2015 import ICDAR2015Dataset
from .TotalText import TotalTextDataset


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def create_dataset(is_train=False) -> Tuple[ds.Dataset, int]:
    config_dict = {
        'backbone': {'backbone_ckpt': './ckpts/resnet50-19c8e357.ckpt',
                     'initializer': 'resnet50',
                     'pretrained': False},
        'ckpt_path': 'ckpts/dbnet_r50.ckpt',
        'config_path': 'config/dbnet/config_resnet50_1p_gpu.yaml',
        'context_mode': 'graph',
        'data_path': '/cache/data',
        'data_url': '',
        'dataset': {'path': 'data',
                    'is_icdar2015': True,
                    'is_show': False,
                    'max_rowsize': 64,
                    'num_workers': 10,
                    'prefetch_size': 10,
                    'short_side': 736,
                    'type': 'ICDAR2015',
                    'train': {
                        'augmentations':
                            {
                                'RandomScale': {'short_side': 736},
                                'RandomRotate': {'random_angle': [-10, 10]},
                                'RandomFlip': {},
                                'RandomCrop':{'max_tries': 10, 'min_crop_side_ratio': 0.1, 'crop_size': (640, 640)}
                            },
                        'transforms': {
                            'MakeSegDetectionData': {'min_text_size': 8, 'shrink_ratio': 0.4},
                            'MakeBorderMap': {'shrink_ratio': 0.4, 'thresh_min': 0.3, 'thresh_max': 0.7}
                        }
                    },
                    'eval': {
                        'transforms': {},
                    }},
        'device_id': 0,
        'device_num': 1,
        'device_target': 'GPU',
        'enable_modelarts': False,
        'eval': {'box_thresh': 0.6,
                 'dest': 'binary',
                 'eval_size': [736, 1280],
                 'gt_dir': './data/test_gts/',
                 'image_dir': './outputs_test/',
                 'img_dir': './data/test_images/',
                 'img_format': '.jpg',
                 'max_candidates': 1000,
                 'polygon': False,
                 'show_images': False,
                 'thresh': 0.3,
                 'unclip_ratio': 1.5},
        'eval_iter': 20,
        'loss': {'bce_replace': 'bceloss', 'bce_scale': 5, 'eps': 1e-06, 'l1_scale': 10},
        'mix_precision': True,
        'net': 'DBnet',
        'optimizer': {'lr': {'base_lr': 0.007, 'factor': 0.9, 'target_lr': 0.0, 'warmup_epoch': 3},
                      'momentum': 0.9,
                      'type': 'sgd',
                      'weight_decay': 0.0001},
        'output_dir': './outputs',
        'rank_id': 0,
        'run_eval': True,
        'seed': 1,
        'segdetector': {'adaptive': True,
                        'bias': False,
                        'in_channels': [256, 512, 1024, 2048],
                        'inner_channels': 256,
                        'k': 50,
                        'serial': False},
        'train': {'batch_size': 20,
                  'dataset_sink_mode': True,
                  'gt_dir': './data/train_gts/',
                  'img_dir': './data/train_images/',
                  'img_format': '.jpg',
                  'is_eval_before_saving': True,
                  'is_transform': True,
                  'log_filename': 'train',
                  'max_checkpoints': 5,
                  'pretrained_ckpt': '',
                  'save_steps': 630,
                  'start_epoch_num': 0,
                  'total_epochs': 1200},
        'train_url': ''}
    config = Config(config_dict)
    """Create MindSpore Dataset object."""
    ds.config.set_prefetch_size(config.dataset.prefetch_size)
    if config.dataset.type == "ICDAR2015":
        augments = config_dict['dataset']['train']['augmentations'] if is_train else {}
        transforms = config_dict['dataset']['train']['transforms'] if is_train else config_dict['dataset']['eval']['transforms']
        data_loader = ICDAR2015Dataset(config.dataset.path, augments, transforms, config.eval.eval_size, train=is_train)
    elif config.dataset.type == "TotalText":
        data_loader = TotalTextDataset(config, isTrain=is_train)
    else:
        raise ValueError(f"Not support dataset.type: {config.dataset.type}.")
    if not hasattr(config, "device_num"):
        config.device_num = 1
    if not hasattr(config, "rank_id"):
        config.rank_id = 0
    if is_train:
        dataset = ds.GeneratorDataset(data_loader,
                                      ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'],
                                      num_parallel_workers=config.dataset.num_workers,
                                      num_shards=config.device_num, shard_id=config.rank_id,
                                      shuffle=True, max_rowsize=config.dataset.max_rowsize)
    else:
        dataset = ds.GeneratorDataset(data_loader, ['img', 'polys', 'ignore'], shuffle=False)
    batch_size = config.train.batch_size if is_train else 1
    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    steps_pre_epoch = dataset.get_dataset_size()
    return dataset, steps_pre_epoch
