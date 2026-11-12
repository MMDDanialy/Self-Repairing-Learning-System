"""Core Localization Metrics Engine for VisDrone Person Evaluation (Checkpoint 4).

Implements vectorized IoU calculations, precision-recall curve analysis, COCO-style mAP@[.50:.95],
geometric box error decompositions (normalized center error and aspect ratio error),
and scale-stratified evaluation across UAV flight envelope target buckets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Represents a predicted bounding box detection."""

    image_id: str
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float
    category_id: int = 0


@dataclass(frozen=True)
class GroundTruth:
    """Represents a ground-truth bounding box annotation."""

    image_id: str
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    category_id: int = 0


@dataclass(frozen=True)
class MatchedPair:
    """Represents an explicitly matched prediction and ground-truth pair."""

    pred: Detection
    gt: GroundTruth
    iou: float
    center_error: float
    aspect_ratio_error: float


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise Intersection-over-Union (IoU) matrix between two sets of boxes.

    Args:
        boxes1: Array of shape (N, 4) in (x1, y1, x2, y2) format.
        boxes2: Array of shape (M, 4) in (x1, y1, x2, y2) format.

    Returns:
        IoU matrix of shape (N, M).
    """
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    # Intersection coordinates
    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Areas
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - intersection
    return np.where(union > 0.0, intersection / union, 0.0)


def compute_center_error(
    pred_box: tuple[float, float, float, float],
    gt_box: tuple[float, float, float, float],
) -> float:
    """Computes the normalized Euclidean center displacement error E_center.

    Formula from CP3/CP4 specification:
        E_center = sqrt((cx_pred - cx_gt)^2 + (cy_pred - cy_gt)^2) / D_diagonal_gt
    """
    cx_p = (pred_box[0] + pred_box[2]) / 2.0
    cy_p = (pred_box[1] + pred_box[3]) / 2.0

    cx_g = (gt_box[0] + gt_box[2]) / 2.0
    cy_g = (gt_box[1] + gt_box[3]) / 2.0

    w_g = gt_box[2] - gt_box[0]
    h_g = gt_box[3] - gt_box[1]
    diag_g = np.sqrt(w_g**2 + h_g**2)

    if diag_g <= 0.0:
        return 0.0

    center_dist = np.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2)
    return float(center_dist / diag_g)


def compute_aspect_ratio_error(
    pred_box: tuple[float, float, float, float],
    gt_box: tuple[float, float, float, float],
) -> float:
    """Computes the aspect-ratio distortion error E_ar.

    Formula:
        E_ar = |(w_pred / h_pred) - (w_gt / h_gt)|
    """
    w_p = pred_box[2] - pred_box[0]
    h_p = pred_box[3] - pred_box[1]
    w_g = gt_box[2] - gt_box[0]
    h_g = gt_box[3] - gt_box[1]

    ar_p = w_p / h_p if h_p > 0.0 else 0.0
    ar_g = w_g / h_g if h_g > 0.0 else 0.0

    return float(abs(ar_p - ar_g))


def compute_precision_recall_ap(
    preds: Sequence[Detection],
    gts: Sequence[GroundTruth],
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Calculates precision, recall, and 101-point interpolated Average Precision at a given IoU threshold."""
    if not gts:
        return {
            "ap": 0.0,
            "tp_count": 0,
            "fp_count": len(preds),
            "precisions": np.array([0.0]),
            "recalls": np.array([0.0]),
            "matched_pairs": [],
        }

    if not preds:
        return {
            "ap": 0.0,
            "tp_count": 0,
            "fp_count": 0,
            "precisions": np.array([0.0]),
            "recalls": np.array([0.0]),
            "matched_pairs": [],
        }

    # Group ground truths by image_id
    gt_by_img: dict[str, list[GroundTruth]] = {}
    for g in gts:
        gt_by_img.setdefault(g.image_id, []).append(g)

    # Sort all detections descending by confidence score
    sorted_preds = sorted(preds, key=lambda p: p.score, reverse=True)

    tp_flags = np.zeros(len(sorted_preds), dtype=np.float64)
    fp_flags = np.zeros(len(sorted_preds), dtype=np.float64)
    assigned_gts: dict[str, set[int]] = {img_id: set() for img_id in gt_by_img}

    matched_pairs: list[MatchedPair] = []

    for idx, p in enumerate(sorted_preds):
        img_gts = gt_by_img.get(p.image_id, [])
        if not img_gts:
            fp_flags[idx] = 1.0
            continue

        p_box = np.array([p.box])
        g_boxes = np.array([g.box for g in img_gts])
        ious = compute_iou_matrix(p_box, g_boxes)[0]

        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= iou_threshold and best_gt_idx not in assigned_gts[p.image_id]:
            tp_flags[idx] = 1.0
            assigned_gts[p.image_id].add(best_gt_idx)

            matched_gt = img_gts[best_gt_idx]
            matched_pairs.append(
                MatchedPair(
                    pred=p,
                    gt=matched_gt,
                    iou=best_iou,
                    center_error=compute_center_error(p.box, matched_gt.box),
                    aspect_ratio_error=compute_aspect_ratio_error(p.box, matched_gt.box),
                )
            )
        else:
            fp_flags[idx] = 1.0

    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum(fp_flags)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / float(len(gts))

    # 101-point COCO interpolation
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = np.zeros_like(recall_thresholds)

    for i, r in enumerate(recall_thresholds):
        prec_subset = precisions[recalls >= r]
        interpolated_precisions[i] = np.max(prec_subset) if len(prec_subset) > 0 else 0.0

    ap = float(np.mean(interpolated_precisions))

    return {
        "ap": ap,
        "tp_count": int(tp_cumsum[-1]),
        "fp_count": int(fp_cumsum[-1]),
        "precisions": precisions,
        "recalls": recalls,
        "matched_pairs": matched_pairs,
    }


def compute_coco_map(
    preds: Sequence[Detection],
    gts: Sequence[GroundTruth],
    start: float = 0.50,
    stop: float = 0.95,
    step: float = 0.05,
) -> dict[str, Any]:
    """Computes COCO-style mAP averaged over IoU thresholds [start:stop:step] (10 thresholds)."""
    thresholds = np.arange(start, stop + 1e-5, step)
    per_thresh: dict[str, float] = {}
    matched_pairs_50: list[MatchedPair] = []
    matched_pairs_75: list[MatchedPair] = []

    ap_values: list[float] = []
    for thresh in thresholds:
        key = f"ap{int(round(thresh * 100))}"
        res = compute_precision_recall_ap(preds, gts, iou_threshold=float(thresh))
        per_thresh[key] = res["ap"]
        ap_values.append(res["ap"])

        if abs(thresh - 0.50) < 1e-4:
            matched_pairs_50 = res["matched_pairs"]
        elif abs(thresh - 0.75) < 1e-4:
            matched_pairs_75 = res["matched_pairs"]

    mean_center_error_75 = (
        float(np.mean([m.center_error for m in matched_pairs_75])) if matched_pairs_75 else 0.0
    )
    mean_ar_error_75 = (
        float(np.mean([m.aspect_ratio_error for m in matched_pairs_75])) if matched_pairs_75 else 0.0
    )

    return {
        "map": float(np.mean(ap_values)),
        "ap50": per_thresh.get("ap50", 0.0),
        "ap75": per_thresh.get("ap75", 0.0),
        "mean_center_error_ap75": mean_center_error_75,
        "mean_aspect_ratio_error_ap75": mean_ar_error_75,
        "matched_count_ap75": len(matched_pairs_75),
        "per_threshold_ap": per_thresh,
    }


def compute_scale_stratified_ap(
    preds: Sequence[Detection],
    gts: Sequence[GroundTruth],
    iou_threshold: float = 0.50,
    scale_buckets: dict[str, tuple[float, float | None]] | None = None,
) -> dict[str, Any]:
    """Computes scale-stratified AP across target area buckets."""
    if scale_buckets is None:
        scale_buckets = {
            "small": (0.0, 1024.0),
            "medium": (1024.0, 9216.0),
            "large_low_alt": (9216.0, None),
        }

    def _box_area(box: tuple[float, float, float, float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    results: dict[str, Any] = {}
    for bucket_name, (min_a, max_a) in scale_buckets.items():
        bucket_gts = [
            g for g in gts if _box_area(g.box) >= min_a and (max_a is None or _box_area(g.box) < max_a)
        ]
        bucket_preds = [
            p for p in preds if _box_area(p.box) >= min_a and (max_a is None or _box_area(p.box) < max_a)
        ]

        ap_res = compute_precision_recall_ap(bucket_preds, bucket_gts, iou_threshold=iou_threshold)
        results[bucket_name] = {
            "ap": ap_res["ap"],
            "gt_count": len(bucket_gts),
            "pred_count": len(bucket_preds),
            "tp_count": ap_res["tp_count"],
            "fp_count": ap_res["fp_count"],
        }

    return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI runner executing demonstration of localization metrics computation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run localization metrics demonstration for Person detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_person.json",
        help="Path to evaluation configuration JSON (default: configs/eval_person.json)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Synthetic demo
    gts = [
        GroundTruth("frame_001", (10.0, 10.0, 50.0, 60.0)),
        GroundTruth("frame_001", (100.0, 100.0, 200.0, 300.0)),
        GroundTruth("frame_002", (20.0, 30.0, 80.0, 120.0)),
    ]
    preds = [
        Detection("frame_001", (11.0, 10.0, 51.0, 60.0), score=0.98),
        Detection("frame_001", (105.0, 102.0, 205.0, 300.0), score=0.89),
        Detection("frame_002", (25.0, 35.0, 75.0, 115.0), score=0.82),
    ]

    coco_metrics = compute_coco_map(preds, gts)
    scale_metrics = compute_scale_stratified_ap(preds, gts, iou_threshold=0.50)

    print("\n" + "=" * 65)
    print("CP4 LOCALIZATION METRICS DEMO")
    print("=" * 65)
    print(f"mAP@[.50:.95]                   : {coco_metrics['map']:.4f}")
    print(f"AP50                            : {coco_metrics['ap50']:.4f}")
    print(f"AP75                            : {coco_metrics['ap75']:.4f}")
    print(f"Mean Center Error @ AP75        : {coco_metrics['mean_center_error_ap75']:.4f}")
    print(f"Mean Aspect Ratio Error @ AP75  : {coco_metrics['mean_aspect_ratio_error_ap75']:.4f}")
    print("\nScale-Stratified AP50:")
    for scale, s_data in scale_metrics.items():
        print(f"  {scale:<15}: AP={s_data['ap']:.4f} (GTs={s_data['gt_count']}, Preds={s_data['pred_count']})")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
