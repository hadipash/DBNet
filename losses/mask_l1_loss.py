from mindspore.nn import LossBase


class MaskL1Loss(LossBase):
    def __init__(self, eps=1e-6):
        super().__init__()
        self._eps = eps

    def construct(self, pred, *labels):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        gt, mask = labels
        pred = pred.squeeze(axis=1)
        return ((pred - gt).abs() * mask).sum() / (mask.sum() + self._eps)
