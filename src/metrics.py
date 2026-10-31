"""Local Panoptic Quality (Kirillov et al., 2019): instances match at IoU > 0.5, PQ = SQ * RQ.
The competition scores the same PQ, but does not document how it aggregates over images: locally TP / FP / FN
are pooled over the whole dataset (`panoptic_quality_dataset`), which gave 0.40 on validation vs 0.34 on the
public leaderboard."""
import numpy as np


def match_instances(gt: np.ndarray, pred: np.ndarray, iou_thresh: float = 0.5) -> dict:
    """Match instances between two label maps (0 = background).
    Returns tp, fp, fn, the summed IoU of matched pairs and the matched pairs."""
    # ids and areas from foreground pixels only: sorting the whole 2048² map is the slow part
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
        "fp": len(pred_area) - tp,
        "fn": len(gt_area) - tp,
        "iou_sum": iou_sum,
        "matches": matches,
    }


def panoptic_quality(gt: np.ndarray, pred: np.ndarray, iou_thresh: float = 0.5) -> dict:
    """PQ, SQ and RQ for a single image."""
    m = match_instances(gt, pred, iou_thresh)
    return pq_from_counts(m["tp"], m["fp"], m["fn"], m["iou_sum"])


def panoptic_quality_dataset(pairs, iou_thresh: float = 0.5) -> dict:
    """PQ accumulated over a dataset: TP/FP/FN and IoU sums are pooled before dividing."""
    tp = fp = fn = 0
    iou_sum = 0.0
    for gt, pred in pairs:
        m = match_instances(gt, pred, iou_thresh)
        tp, fp, fn, iou_sum = tp + m["tp"], fp + m["fp"], fn + m["fn"], iou_sum + m["iou_sum"]
    return pq_from_counts(tp, fp, fn, iou_sum)


def pq_from_counts(tp: int, fp: int, fn: int, iou_sum: float) -> dict:
    denom = tp + 0.5 * fp + 0.5 * fn
    if denom == 0:  # nothing in either map
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}
    sq = iou_sum / tp if tp else 0.0
    rq = tp / denom
    return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


def error_types(gt: np.ndarray, pred: np.ndarray, iou_thresh: float = 0.5) -> dict:
    """Why instances fail to match. Each unmatched GT instance is
    missed (touches no prediction), merged (a prediction it touches also touches another GT instance),
    split (touches several predictions) or boundary (one prediction, IoU <= thresh).
    Unmatched predictions touching no GT instance are spurious."""
    m = match_instances(gt, pred, iou_thresh)
    matched_g = {g for g, _, _ in m["matches"]}
    matched_p = {p for _, p, _ in m["matches"]}
    both = (gt != 0) & (pred != 0)
    pairs = np.unique(np.stack([gt[both], pred[both]]), axis=1).T if both.any() else np.zeros((0, 2), int)
    preds_of, gts_of = {}, {}
    for g, p in pairs:
        preds_of.setdefault(int(g), set()).add(int(p))
        gts_of.setdefault(int(p), set()).add(int(g))

    out = {"matched": len(matched_g), "missed": 0, "merged": 0, "split": 0, "boundary": 0, "spurious": 0}
    for g in np.unique(gt[gt != 0]):
        if int(g) in matched_g:
            continue
        ps = preds_of.get(int(g), set())
        if not ps:
            out["missed"] += 1
        elif any(len(gts_of[p]) > 1 for p in ps):
            out["merged"] += 1
        elif len(ps) > 1:
            out["split"] += 1
        else:
            out["boundary"] += 1
    out["spurious"] = sum(int(p) not in matched_p and int(p) not in gts_of for p in np.unique(pred[pred != 0]))
    return out
