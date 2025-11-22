"""Metrics for temporal localization."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

from .model_interface import TimestampPrediction


def compute_iou(pred: TimestampPrediction, target: TimestampPrediction) -> float:
    """Compute temporal Intersection-over-Union in seconds."""

    pred_start, pred_end = pred
    tgt_start, tgt_end = target

    inter_start = max(pred_start, tgt_start)
    inter_end = min(pred_end, tgt_end)
    intersection = max(0.0, inter_end - inter_start)

    union = max(pred_end, tgt_end) - min(pred_start, tgt_start)
    if union == 0:
        return 0.0
    return intersection / union


def summarize_iou(
    ious: Sequence[float], thresholds: Sequence[float] = (0.3, 0.5, 0.7)
) -> Mapping[str, float]:
    if not ious:
        return {"mIoU": 0.0, **{f"IoU@{t}": 0.0 for t in thresholds}}

    m_iou = sum(ious) / len(ious)
    summary = {"mIoU": m_iou}
    for t in thresholds:
        passed = sum(1 for val in ious if val >= t)
        summary[f"IoU@{t}"] = passed / len(ious)
    return summary


EvaluationFields = Tuple[str, TimestampPrediction, TimestampPrediction, float]


def results_to_rows(results: Iterable[EvaluationFields]):
    for qid, pred, target, iou in results:
        yield {
            "qid": qid,
            "pred": pred,
            "target": target,
            "iou": iou,
        }
