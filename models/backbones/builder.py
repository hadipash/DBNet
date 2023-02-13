from typing import Union

from mindcv.models import load_pretrained
from mindcv.models.resnet import Bottleneck, default_cfgs

from .resnet_dcn import DBNetResNetDCN, DBNetResNet


def create_backbone(name: str, deform_conv=False, pretrained: Union[bool, str] = True, **kwargs):
    if name == 'resnet50':
        layers = [3, 4, 6, 3]
        network = DBNetResNetDCN(Bottleneck, layers) if deform_conv else DBNetResNet(Bottleneck, layers)
    else:
        raise ValueError(f'Not supported backbone: {name}')

    if pretrained:
        load_pretrained(network, default_cfgs[name] if isinstance(pretrained, bool) else pretrained, num_classes=0)

    return network
