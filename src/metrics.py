"""Local Panoptic Quality (Kirillov et al., 2019): instances match at IoU > 0.5, PQ = SQ * RQ.
TODO: confirm against the Kaggle Evaluation page (aggregation over images, empty-image handling)."""
import numpy as np


def match_instances(gt: np.ndarray, pred: np.ndarray, iou_thresh: float = 0.5) -> dict:
    """Match instances between two label maps (0 = background).
    Returns tp, fp, fn, the summed IoU of matched pairs and the matched pairs."""
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids != 0]

    gt_area = {int(i): int(a) for i, a in zip(*np.unique(gt[gt != 0], return_counts=True))}
    pred_area = {int(i): int(a) for i, a in zip(*np.unique(pred[pred != 0], return_counts=True))}

    # Intersections, only over pixels where both maps are foreground
    both = (gt != 0) & (pred != 0)
    pairs, inter = np.unique(np.stack([gt[both], pred[both]]), axis=1, return_counts=True)

    matches, iou_sum = [], 0.0
    for (g, p), n in zip(pairs.T, inter):
        iou = float(n / (gt_area[int(g)] + pred_area[int(p)] - n))
        if iou > iou_thresh:
            matches.append((int(g), int(p), float(iou)))
            iou_sum += iou

    tp = len(matches)
    return {
        "tp": tp,
        "fp": len(pred_ids) - tp,
        "fn": len(gt_ids) - tp,
        "iou_sum": iou_sum,
        "matches": matches,
    }


def panoptic_quality(gt: np.ndarray, pred: np.ndarray, iou_thresh: float = 0.5) -> dict:
    """PQ, SQ and RQ for a single image."""
    m = match_instances(gt, pred, iou_thresh)
    return _pq_from_counts(m["tp"], m["fp"], m["fn"], m["iou_sum"])


def panoptic_quality_dataset(pairs, iou_thresh: float = 0.5) -> dict:
    """PQ accumulated over a dataset: TP/FP/FN and IoU sums are pooled before dividing."""
    tp = fp = fn = 0
    iou_sum = 0.0
    for gt, pred in pairs:
        m = match_instances(gt, pred, iou_thresh)
        tp, fp, fn, iou_sum = tp + m["tp"], fp + m["fp"], fn + m["fn"], iou_sum + m["iou_sum"]
    return _pq_from_counts(tp, fp, fn, iou_sum)


def _pq_from_counts(tp: int, fp: int, fn: int, iou_sum: float) -> dict:
    denom = tp + 0.5 * fp + 0.5 * fn
    if denom == 0:  # nothing in either map
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}
    sq = iou_sum / tp if tp else 0.0
    rq = tp / denom
    return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}
