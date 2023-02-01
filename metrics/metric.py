import numpy as np
from shapely.geometry import Polygon

from mindspore import nn

from .post_process import SegDetectorRepresenter


def _get_intersect(pD, pG):
    return pD.intersection(pG).area


def _get_iou(pD, pG):
    return pD.intersection(pG).area / pD.union(pG).area


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DetectionIoUEvaluator:
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt: list, preds: list):
        # filter invalid groundtruth polygons and split them into useful and ignored
        gt_polys, ignore_gt_polys = [], []
        for sample in gt:
            poly = Polygon(sample['polys'])
            if poly.is_valid and poly.is_simple:
                if not sample['dontcare']:
                    gt_polys.append(poly)
                else:
                    ignore_gt_polys.append(poly)

        # repeat the same step for the predicted polygons
        det_polys, ignore_det_polys = [], []
        for pred in preds:
            poly = Polygon(pred)
            if poly.is_valid and poly.is_simple:
                poly_area = poly.area
                if ignore_gt_polys and poly_area > 0:
                    for ignore_poly in ignore_gt_polys:
                        intersected_area = _get_intersect(ignore_poly, poly)
                        precision = intersected_area / poly_area
                        # If precision enough, append as ignored detection
                        if precision > self.area_precision_constraint:
                            ignore_det_polys.append(poly)
                            break
                    else:
                        det_polys.append(poly)
                else:
                    det_polys.append(poly)

        pairs = []
        det_match = 0
        iou_mat = np.empty([1, 1])
        if gt_polys and det_polys:
            iou_mat = np.empty([len(gt_polys), len(det_polys)])
            det_rect_mat = np.zeros(len(det_polys), np.int8)

            for gt_idx in range(len(gt_polys)):
                for det_idx in range(len(det_polys)):
                    if det_rect_mat[det_idx] == 0:  # the match is not found yet
                        iou_mat[gt_idx, det_idx] = _get_iou(det_polys[det_idx], gt_polys[gt_idx])
                        if iou_mat[gt_idx, det_idx] > self.iou_constraint:
                            # Mark the visit arrays
                            det_rect_mat[det_idx] = 1
                            det_match += 1
                            pairs.append({'gt': gt_idx, 'det': det_idx})
                            break

        if not gt_polys:
            recall = 1.0
            precision = 0.0 if det_polys else 1.0
        else:
            recall = float(det_match) / len(gt_polys)
            precision = float(det_match) / len(det_polys) if det_polys else 0
        hmean = 0 if (precision + recall) == 0 else \
                2.0 * precision * recall / (precision + recall)

        metric = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iou_mat': [] if len(det_polys) > 100 else iou_mat.tolist(),
            'gt_polys': gt_polys,
            'det_polys': det_polys,
            'gt_care_num': len(gt_polys),
            'det_care_num': len(det_polys),
            'gt_dont_care': ignore_gt_polys,
            'det_dont_care': ignore_det_polys,
            'det_matched': det_match
        }
        return metric

    def combine_results(self, results):
        num_global_care_gt = 0
        num_global_care_det = 0
        matched_sum = 0
        for result in results:
            num_global_care_gt += result['gt_care_num']
            num_global_care_det += result['det_care_num']
            matched_sum += result['det_matched']

        method_recall = 0 if num_global_care_gt == 0 else float(
            matched_sum) / num_global_care_gt
        method_precision = 0 if num_global_care_det == 0 else float(
            matched_sum) / num_global_care_det
        methodHmean = 0 if method_recall + method_precision == 0 else 2 * method_recall * method_precision / \
                                                                      (method_recall + method_precision)
        method_metrics = {'precision': method_precision,
                          'recall': method_recall, 'hmean': methodHmean}
        return method_metrics


class QuadMetric:
    def __init__(self, is_output_polygon=False):
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator()

    def measure(self, batch, output, box_thresh=0.7):
        '''
        batch: (image, polygons, ignore_tags)
            image: numpy array of shape (N, C, H, W).
            polys: numpy array of shape (N, K, 4, 2), the polygons of objective regions.
            dontcare: numpy array of shape (N, K), indicates whether a region is ignorable or not.
        output: (polygons, ...)
        '''
        gt_polys = batch['polys'].astype(np.float32)
        gt_dontcare = batch['dontcare']
        pred_polys = np.array(output[0])
        pred_scores = np.array(output[1])

        # Loop i for every batch
        for i in range(len(gt_polys)):  # FIXME: iterates over all samples in a batch, but evaluates only the last one
            gt = [{'polys': gt_polys[i][j], 'dontcare': gt_dontcare[i][j]}
                  for j in range(len(gt_polys[i]))]
            if self.is_output_polygon:
                pred = [pred_polys[i][j] for j in range(len(pred_polys[i]))]    # TODO: why polygons are not filtered?
            else:
                pred = [pred_polys[i][j, :, :].astype(np.int32)
                        for j in range(pred_polys[i].shape[0]) if pred_scores[i][j] >= box_thresh]
        return self.evaluator.evaluate_image(gt, pred)


    def validate_measure(self, batch, output):
        return self.measure(batch, output, box_thresh=0.55)     # TODO: why is here a fixed threshold and different from the above?

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics for image_metrics in raw_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }


class MyMetric(nn.Metric):
    def __init__(self):
        super().__init__()
        self._post_process = SegDetectorRepresenter(box_thresh=0.6)
        self.clear()

    def clear(self):
        self._metric = QuadMetric()
        self._raw_metrics = []

    def update(self, *inputs):
        preds, gt = inputs
        gt = {'polys': gt[0].asnumpy(), 'dontcare': gt[1].asnumpy()}    # FIXME
        boxes, scores = self._post_process(preds)
        self._raw_metrics.append(self._metric.validate_measure(gt, (boxes, scores)))

    def eval(self):
        return self._metric.gather_measure(self._raw_metrics)
