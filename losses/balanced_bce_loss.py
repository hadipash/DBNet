import mindspore as ms
from mindspore import nn, ops


class BalancedCrossEntropyLoss(nn.LossBase):
    """Balanced cross entropy loss."""
    def __init__(self, negative_ratio=3, eps=1e-6):
        super().__init__()
        self._negative_ratio = negative_ratio
        self._eps = eps
        self._bce_loss = ops.BinaryCrossEntropy(reduction='none')

    def construct(self, logits, *labels):
        gt, mask = labels

        logits = logits.squeeze(axis=1)
        gt = gt.squeeze(axis=1)

        positive = gt * mask
        negative = (1 - gt) * mask

        pos_count = positive.sum().astype(ms.int32)
        neg_count = ops.minimum(negative.sum(), pos_count * self._negative_ratio + 1).astype(ms.int32)  # FIXME: + 1 when pos_count is 0

        loss = self._bce_loss(logits, gt, None)

        pos_loss = loss * positive
        neg_loss = loss * negative

        neg_loss, _ = ops.top_k(neg_loss.view(-1), int(neg_count))

        return (pos_loss.sum() + neg_loss.sum()) / (pos_count + neg_count + self._eps)
