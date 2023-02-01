from mindspore import nn

from .balanced_bce_loss import BalancedCrossEntropyLoss
from .dice_loss import DiceLoss
from .mask_l1_loss import MaskL1Loss


class L1BalanceCELoss(nn.LossBase):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, bce_scale=5, l1_scale=10, bce_replace="bceloss"):
        super(L1BalanceCELoss, self).__init__()

        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()

        if bce_replace == "bceloss":
            self.bce_loss = BalancedCrossEntropyLoss()
        elif bce_replace == "diceloss":
            self.bce_loss = DiceLoss()
        else:
            raise ValueError(f"bce_replace should be in ['bceloss', 'diceloss'], but get {bce_replace}")

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def construct(self, pred, labels):
        """
        pred: A dict which contains predictions.
            thresh: The threshold prediction
            binary: The text segmentation prediction.
            thresh_binary: Value produced by `step_function(binary - thresh)`.
        gt: Text regions bitmap gt.
        mask: Ignore mask, pexels where value is 1 indicates no contribution to loss.
        thresh_mask: Mask indicates regions cared by thresh supervision.
        thresh_map: Threshold gt.
        """
        gt, gt_mask, thresh_map, thresh_mask = labels
        bce_loss_output = self.bce_loss(pred['binary'], gt, gt_mask)

        if 'thresh' in pred:
            l1_loss = self.l1_loss(pred['thresh'], thresh_map, thresh_mask)
            dice_loss = self.dice_loss(pred['thresh_binary'], gt, gt_mask)
            loss = dice_loss + self.l1_scale * l1_loss + self.bce_scale * bce_loss_output
        else:
            loss = bce_loss_output

        return loss
