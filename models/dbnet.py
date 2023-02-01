from typing import Tuple

from mindspore import nn, ops, Tensor


def resize_nn(x: Tensor, scale: int = 0, shape: Tuple[int] = None):
    if scale == 1 or shape == x.shape[2:]:
        return x

    if scale:
        shape = (x.shape[2] * scale, x.shape[3] * scale)
    return ops.ResizeNearestNeighbor(shape)(x)


class AdaptiveScaleFusion(nn.Cell):
    # TODO: completed, just move fuse layer in model printout to correct position
    def __init__(self, in_channels, channel_attention=True):
        super().__init__()
        out_channels = in_channels // 4
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)

        self.chan_att = nn.SequentialCell([
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, pad_mode='valid'),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, pad_mode='valid'),
            nn.Sigmoid()
        ]) if channel_attention else None

        self.spat_att = nn.SequentialCell([
            nn.Conv2d(1, 1, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1, pad_mode='valid'),
            nn.Sigmoid()
        ])

        self.scale_att = nn.SequentialCell([
            nn.Conv2d(out_channels, 4, kernel_size=1, pad_mode='valid'),
            nn.Sigmoid()
        ])

    def construct(self, x):
        reduced = self.conv(ops.concat(x, axis=1))

        if self.chan_att is not None:
            ada_pool = ops.mean(reduced, axis=(-2, -1), keep_dims=True)     # equivalent to nn.AdaptiveAvgPool2d(1)
            reduced = self.chan_att(ada_pool) + reduced

        spatial = ops.mean(reduced, axis=1, keep_dims=True)
        spat_att = self.spat_att(spatial) + reduced

        scale_att = self.scale_att(spat_att)
        return ops.concat([scale_att[:, i:i+1] * x[i] for i in range(len(x))], axis=1)


class DBNet(nn.Cell):
    def __init__(self, backbone, in_channels, inner_channels=256, bias=False, adaptive=False, k=50):
        super().__init__()
        self.backbone = backbone
        self.adaptive = adaptive
        self.inner_channels = inner_channels

        assert len(in_channels) == 4, f'Number of input features should be 4, instead received {len(in_channels)}'
        self.unify_channels = nn.CellList([nn.Conv2d(ch, inner_channels, kernel_size=1, has_bias=bias, pad_mode='valid')
                                           for ch in in_channels])

        outer_channels = inner_channels // 4
        self.out = nn.CellList([nn.Conv2d(inner_channels, outer_channels, kernel_size=3, padding=1, pad_mode='pad',
                                          has_bias=bias) for _ in range(4)])

        self.fuse = ops.Concat(axis=1)

        self.segm = self._init_heatmap(inner_channels, outer_channels, bias)
        if adaptive:
            self.thresh = self._init_heatmap(inner_channels, outer_channels, bias)
            self.k = k
            self.diff_bin = nn.Sigmoid()

    @staticmethod
    def _init_heatmap(inner_channels, outer_channels, bias):
        return nn.SequentialCell([     # `pred` block from the original work
            nn.Conv2d(inner_channels, outer_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=bias),
            nn.BatchNorm2d(outer_channels),
            nn.ReLU(),
            nn.Conv2dTranspose(outer_channels, outer_channels, kernel_size=2, stride=2, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(outer_channels),
            nn.ReLU(),
            nn.Conv2dTranspose(outer_channels, 1, kernel_size=2, stride=2, pad_mode='valid', has_bias=True),
            nn.Sigmoid()
        ])

    def construct(self, x: Tensor) -> dict:
        x = list(self.backbone(x))

        for i, uc_op in enumerate(self.unify_channels):
            x[i] = uc_op(x[i])

        for i in range(2, -1, -1):
            x[i] += resize_nn(x[i+1], shape=x[i].shape[2:])

        for i, out in enumerate(self.out):
            x[i] = resize_nn(out(x[i]), shape=x[0].shape[2:])

        fuse = self.fuse(x[::-1])  # matching the reverse order of the original work

        pred = {'binary': self.segm(fuse)}

        if self.adaptive:
            pred['thresh'] = self.thresh(fuse)
            pred['thresh_binary'] = self.diff_bin(self.k * (pred['binary'] - pred['thresh']))   # Differentiable Binarization

        return pred


class DBNetPP(DBNet):
    # TODO: parameters passing
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fuse = AdaptiveScaleFusion(self.inner_channels)


def create_model(model: str, backbone='deform_resnet50', train=False):
    from backbones.resnet_dcn import DBNetResNet, DBNetResNetDCN, Bottleneck

    backbone = DBNetResNetDCN(Bottleneck, [3, 4, 6, 3])

    if model == 'dbnet':
        return DBNet(backbone, in_channels=[256, 512, 1024, 2048], adaptive=train)
    elif model == 'dbnet++':
        return DBNetPP(backbone=backbone, in_channels=[256, 512, 1024, 2048], adaptive=train)
    else:
        raise ValueError(f'Unknown model {model}')
