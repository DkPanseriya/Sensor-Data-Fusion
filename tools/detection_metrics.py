"""COCO-style detection metrics: greedy matching by confidence, 101-point interpolated AP."""
import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)


def box_iou(a, b):
    """IoU matrix between xyxy boxes a (N, 4) and b (M, 4)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(pred_boxes, pred_scores, gt_boxes, thresholds=IOU_THRESHOLDS):
    """TP flags (P, T) for one image and one class: each prediction, highest score first,
    takes the unclaimed ground truth with the highest IoU, if that IoU reaches the threshold."""
    tp = np.zeros((len(pred_boxes), len(thresholds)), bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp
    iou = box_iou(pred_boxes, gt_boxes)
    order = np.argsort(-np.asarray(pred_scores))
    for t, thr in enumerate(thresholds):
        claimed = np.zeros(len(gt_boxes), bool)
        for p in order:
            candidates = np.where(~claimed, iou[p], -1.0)
            g = int(np.argmax(candidates))
            if candidates[g] >= thr:
                tp[p, t] = True
                claimed[g] = True
    return tp


def average_precision(tp, scores, n_gt):
    """AP per threshold (T,) from TP flags (P, T) pooled over images. NaN when the class has no ground truth."""
    if n_gt == 0:
        return np.full(tp.shape[1], np.nan)
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = tp[order]
    tp_cum = np.cumsum(tp, axis=0)
    fp_cum = np.cumsum(~tp, axis=0)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    grid = np.linspace(0, 1, 101)
    ap = np.zeros(tp.shape[1])
    for t in range(tp.shape[1]):
        envelope = np.maximum.accumulate(precision[::-1, t])[::-1] if len(tp) else np.zeros(0)
        idx = np.searchsorted(recall[:, t], grid, side="left")
        ap[t] = np.mean([envelope[i] if i < len(envelope) else 0.0 for i in idx])
    return ap


def evaluate(predictions, ground_truth, n_classes, thresholds=IOU_THRESHOLDS):
    """predictions / ground_truth: dict image -> (classes (N,), boxes (N, 4)[, scores (N,)]).

    Returns AP array (classes, thresholds), NaN rows for classes without ground truth.
    """
    ap = np.full((n_classes, len(thresholds)), np.nan)
    for c in range(n_classes):
        flags, scores, n_gt = [], [], 0
        for image, (g_cls, g_box) in ground_truth.items():
            gt = g_box[g_cls == c]
            n_gt += len(gt)
            p_cls, p_box, p_score = predictions.get(image, (np.zeros(0), np.zeros((0, 4)), np.zeros(0)))
            keep = p_cls == c
            flags.append(match_image(p_box[keep], p_score[keep], gt, thresholds))
            scores.append(p_score[keep])
        ap[c] = average_precision(np.vstack(flags) if flags else np.zeros((0, len(thresholds)), bool),
                                  np.concatenate(scores) if scores else np.zeros(0), n_gt)
    return ap
