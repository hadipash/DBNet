import mindspore as ms
from mindspore import nn, ops, numpy as mnp


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

        pos_count = positive.sum(axis=(1, 2), keepdims=True).astype(ms.int32)
        neg_count = negative.sum(axis=(1, 2), keepdims=True).astype(ms.int32)

        neg_count = ops.minimum(neg_count, pos_count * self._negative_ratio + 1).squeeze(axis=(1, 2))   # FIXME: + 1 when pos_count is 0

        loss = self._bce_loss(logits, gt, None)

        pos_loss = loss * positive
        neg_loss = (loss * negative).view(loss.shape[0], -1)

        neg_vals, _ = ops.sort(neg_loss)
        neg_index = ops.stack((mnp.arange(loss.shape[0]), neg_vals.shape[1] - neg_count), axis=1)
        min_neg_score = ops.expand_dims(ops.gather_nd(neg_vals, neg_index), axis=1)

        neg_loss_mask = (neg_loss >= min_neg_score).astype(ms.float32)  # filter values less than top k
        neg_loss_mask = ops.stop_gradient(neg_loss_mask)

        neg_loss = neg_loss_mask * neg_loss

        return (pos_loss.sum() + neg_loss.sum()) / \
               (pos_count.astype(ms.float32).sum() + neg_count.astype(ms.float32).sum() + self._eps)
