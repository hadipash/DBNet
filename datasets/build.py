from mindspore import dataset as ds

from .ICDAR2015 import ICDAR2015Dataset
from .SynthText import SynthTextDataset
from .TotalText import TotalTextDataset


def create_dataset(config: dict, stage='train') -> ds.Dataset:
    if config['name'] == "ICDAR2015":
        data_class = ICDAR2015Dataset
    elif config['name'] == "TotalText":
        data_class = TotalTextDataset
    elif config['name'] == "SynthText":
        data_class = SynthTextDataset
    else:
        raise ValueError(f"Not supported dataset: {config['name']}.")

    ds.config.set_prefetch_size(config['prefetch_size'])
    dataset = data_class(config['path'], config[stage]['augmentations'], config[stage]['transforms'],
                         config['val']['size'] if stage != 'train' else None, train=(stage == 'train'))

    dataset = ds.GeneratorDataset(dataset, config[stage]['keys'], num_parallel_workers=config[stage]['num_workers'],
                                  shuffle=(stage == 'train'), max_rowsize=16)
    return dataset.batch(config[stage]['batch_size'])
