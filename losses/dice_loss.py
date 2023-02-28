from mindspore import nn


class DiceLoss(nn.LossBase):
    def __init__(self, eps=1e-6):
        super().__init__()
        self._eps = eps

    def construct(self, pred, *labels):
        """
        pred: one or two heatmaps of shape (N, 1, H, W),
              the losses of two heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        """
        gt, mask = labels
        pred = pred.squeeze(axis=1) * mask
        gt = gt.squeeze(axis=1) * mask

        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() + self._eps
        return 1 - 2.0 * intersection / union
