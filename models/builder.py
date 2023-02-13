from mindspore import load_checkpoint
from .backbones.builder import create_backbone
from .dbnet import DBNet, DBNetPP


def create_model(model: dict):
    model['backbone'] = create_backbone(**model['backbone'])

    if model['name'] == 'dbnet':
        network = DBNet
    elif model['name'] == 'dbnet++':
        network = DBNetPP
    else:
        raise ValueError(f'Unknown model {model}')

    network = network(**model)
    if model['load_ckpt']:
        load_checkpoint(model['load_ckpt'], network)

    return network
