from typing import Tuple

from mindspore import nn, ops, Tensor
from mindcv.models.resnet import ResNet, Bottleneck


# TODO:
# 1. reduced lr for offset_conv in DeformConv2d (the added conv layers for offset and modulation learning
# are set to 0.1 times those of the existing layers).
# 2. check dilation steps in DeformConv2d


class DeformConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, group: int = 1, has_bias: bool = False, deform_group: int = 1, modulated: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, group=group,
                         has_bias=False, weight_init='normal', bias_init='zeros')

        self.stride = (1, 1, stride, stride)
        self.padding = (padding,) * 4
        self.dilation = (1, 1, dilation, dilation)
        self.deform_group = deform_group
        self.modulated = modulated

        # FIXME: weight initialization here may cause a problem on Ascend architecture,
        # namely it should be a float number with the fractional part, i.e. `numpy.ones()` will not work.
        self.offset_conv = nn.Conv2d(in_channels, 3 * deform_group * (kernel_size ** 2), kernel_size,
                                     stride, pad_mode='pad', padding=self.padding, has_bias=has_bias,
                                     weight_init='zeros')

    def construct(self, x: Tensor) -> Tensor:
        offset = self.offset_conv(x)
        # TODO: Does the mask lies in the range [0, 1]? Mask is a  modulation mechanism ∆m
        return ops.deformable_conv2d(x, self.weight, offset, self.kernel_size, self.stride, self.padding,
                                     self.bias, self.dilation, self.group, self.deform_group, self.modulated)


class BottleneckDCN(Bottleneck):
    """
    ResNet Bottleneck with 3x3 2D convolution replaced by
    `Modulated Deformable Convolution <https://arxiv.org/abs/1811.11168>`__ .
    """
    def __init__(self, in_channels: int, channels: int, stride: int = 1, groups: int = 1, base_width: int = 64,
                 **kwargs):
        super().__init__(in_channels, channels, stride, groups, base_width, **kwargs)

        width = int(channels * (base_width / 64.0)) * groups
        self.conv2 = DeformConv2d(width, width, kernel_size=3, stride=stride, padding=1, group=groups)


class DBNetResNet(ResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)
        self.out_channels = [256, 512, 1024, 2048]  # TODO: calculate automatically
        del self.pool, self.classifier  # remove the original header to avoid confusion

    def construct(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x2, x3, x4, x5


class DBNetResNetDCN(DBNetResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)

        self.input_channels = 64 * block.expansion  # reset the input channels counter. TODO: fix it?
        self.layer2 = self._make_layer(BottleneckDCN, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BottleneckDCN, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BottleneckDCN, 512, layers[3], stride=2)
