"""Localization Distillation (LD) Loss Engine for Person Detection (Checkpoint 4).

Implements numerically stable temperature-scaled softmax, log-softmax, discrete bounding-box
edge probability distribution modeling, KL-divergence Localization Distillation loss, and
Valuable Localization Region (VLR) masking following CVPR 2022 Localization Distillation.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


def softmax(logits: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax with temperature scaling.

    Args:
        logits: Input logit array of any shape.
        temperature: Temperature parameter tau > 0 (default: 1.0).
        axis: Dimension along which softmax is computed (default: -1).

    Returns:
        Probability distribution array summing to 1.0 along specified axis.
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")

    scaled = logits / float(temperature)
    max_val = np.max(scaled, axis=axis, keepdims=True)
    exp_logits = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_logits, axis=axis, keepdims=True)
    return exp_logits / np.maximum(sum_exp, 1e-12)


def log_softmax(logits: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax with temperature scaling.

    Args:
        logits: Input logit array of any shape.
        temperature: Temperature parameter tau > 0 (default: 1.0).
        axis: Dimension along which log-softmax is computed (default: -1).

    Returns:
        Log-probabilities array along specified axis.
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")

    scaled = logits / float(temperature)
    max_val = np.max(scaled, axis=axis, keepdims=True)
    stabilized = scaled - max_val
    log_sum_exp = np.log(np.sum(np.exp(stabilized), axis=axis, keepdims=True) + 1e-12)
    return stabilized - log_sum_exp


def compute_vlr_mask(
    anchor_points_xy: np.ndarray,
    target_boxes_xyxy: np.ndarray,
    margin_ratio: float = 0.25,
) -> np.ndarray:
    """Computes Valuable Localization Region (VLR) mask for candidate anchor centers.

    Expands target boxes by margin_ratio * [width, height] in all four directions.
    Anchors falling within any expanded target box are marked True.

    Args:
        anchor_points_xy: Array of shape (A, 2) containing (x, y) anchor center coordinates.
        target_boxes_xyxy: Array of shape (B, 4) containing [x1, y1, x2, y2] bounding boxes.
        margin_ratio: Relative expansion factor around target boxes (default: 0.25).

    Returns:
        Boolean mask array of shape (A,) indicating membership in Valuable Localization Region.
    """
    n_anchors = anchor_points_xy.shape[0]
    n_targets = target_boxes_xyxy.shape[0]

    if n_anchors == 0:
        return np.zeros(0, dtype=bool)
    if n_targets == 0:
        return np.zeros(n_anchors, dtype=bool)

    # Compute expanded VLR boxes
    widths = target_boxes_xyxy[:, 2] - target_boxes_xyxy[:, 0]
    heights = target_boxes_xyxy[:, 3] - target_boxes_xyxy[:, 1]
    dx = margin_ratio * widths
    dy = margin_ratio * heights

    vlr_x1 = target_boxes_xyxy[:, 0] - dx
    vlr_y1 = target_boxes_xyxy[:, 1] - dy
    vlr_x2 = target_boxes_xyxy[:, 2] + dx
    vlr_y2 = target_boxes_xyxy[:, 3] + dy

    # Vectorized containment test: (A, 1) vs (1, B)
    px = anchor_points_xy[:, 0:1]
    py = anchor_points_xy[:, 1:2]

    in_x = (px >= vlr_x1[None, :]) & (px <= vlr_x2[None, :])
    in_y = (py >= vlr_y1[None, :]) & (py <= vlr_y2[None, :])

    return np.any(in_x & in_y, axis=1)


def compute_ld_loss(
    student_edge_logits: np.ndarray,
    teacher_edge_logits: np.ndarray,
    box_weights: np.ndarray | None = None,
    temperature: float = 10.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> np.ndarray | float:
    """Computes Localization Distillation (LD) loss over discrete bounding-box edge distributions.

    Implements CVPR 2022 Localization Distillation:
        L_edge = tau^2 * sum_{k=0}^{K-1} P_T(k) * (log P_T(k) - log P_S(k))
        L_box  = sum_{e in {l, t, r, b}} L_edge(e)

    Args:
        student_edge_logits: Student logits of shape (N, 4, num_bins).
        teacher_edge_logits: Teacher logits of shape (N, 4, num_bins).
        box_weights: Optional per-box/per-anchor weighting array of shape (N,).
        temperature: Softmax distillation temperature tau (default: 10.0).
        reduction: Reduction mode: 'mean', 'sum', or 'none' (default: 'mean').

    Returns:
        Scalar float loss for 'mean' / 'sum', or 1D array of shape (N,) for 'none'.
    """
    if student_edge_logits.shape != teacher_edge_logits.shape:
        raise ValueError(
            f"Shape mismatch: student {student_edge_logits.shape} vs teacher {teacher_edge_logits.shape}"
        )

    n_boxes = student_edge_logits.shape[0]
    if n_boxes == 0:
        if reduction == "none":
            return np.zeros(0, dtype=np.float64)
        return 0.0

    # Compute temperature-scaled teacher probabilities and log-probabilities
    p_teacher = softmax(teacher_edge_logits, temperature=temperature, axis=-1)
    log_p_teacher = log_softmax(teacher_edge_logits, temperature=temperature, axis=-1)
    log_p_student = log_softmax(student_edge_logits, temperature=temperature, axis=-1)

    # KL Divergence: P_T * (log P_T - log P_S) across bins
    # Shape: (N, 4, num_bins)
    kl_per_bin = p_teacher * (log_p_teacher - log_p_student)

    # Sum across distribution bins: shape (N, 4)
    kl_per_edge = np.sum(kl_per_bin, axis=-1)

    # Guard against minor negative values from floating point roundoff
    kl_per_edge = np.maximum(0.0, kl_per_edge)

    # Multiply by tau^2 and sum across 4 edges (left, top, right, bottom): shape (N,)
    tau_sq = float(temperature**2)
    loss_per_box = tau_sq * np.sum(kl_per_edge, axis=-1)

    # Apply per-box weights if provided
    if box_weights is not None:
        if box_weights.shape[0] != n_boxes:
            raise ValueError(
                f"box_weights length {box_weights.shape[0]} does not match boxes {n_boxes}"
            )
        weighted_loss = loss_per_box * box_weights
    else:
        weighted_loss = loss_per_box

    if reduction == "none":
        return weighted_loss
    elif reduction == "sum":
        return float(np.sum(weighted_loss))
    elif reduction == "mean":
        if box_weights is not None:
            sum_weights = float(np.sum(box_weights))
            return float(np.sum(weighted_loss) / (sum_weights + 1e-12)) if sum_weights > 0 else 0.0
        return float(np.mean(weighted_loss))
    else:
        raise ValueError(f"Unknown reduction mode: {reduction}. Choose 'mean', 'sum', or 'none'.")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI runner demonstrating Localization Distillation loss and teacher-student diagnostic."""
    import argparse
    import json
    from src.evaluation.localization_metrics import Detection, GroundTruth
    from src.evaluation.tide_diagnostic import run_tide_diagnostics
    from src.models.base_detector import DeterministicMockDetector

    parser = argparse.ArgumentParser(
        description="Run Localization Distillation (LD) loss and diagnostic demonstration."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Distillation temperature tau (default: 10.0)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 65)
    print("CP4 LOCALIZATION DISTILLATION (LD) ENGINE DEMONSTRATION")
    print("=" * 65)

    # 1. Teacher and Student Mock Detectors
    teacher = DeterministicMockDetector(seed=42, perturbation_mode="none")
    student_loc_biased = DeterministicMockDetector(seed=84, perturbation_mode="loc_biased")

    mock_samples = [
        type("Sample", (), {
            "stem": f"frame_{i:03d}",
            "boxes": [(100.0 + i * 20, 100.0 + i * 15, 180.0 + i * 20, 260.0 + i * 15)],
        })()
        for i in range(5)
    ]

    t_out = teacher.extract_distillation_outputs(mock_samples)
    s_out = student_loc_biased.extract_distillation_outputs(mock_samples)

    t_logits = np.concatenate([o.edge_logits for o in t_out], axis=0)
    s_logits = np.concatenate([o.edge_logits for o in s_out], axis=0)

    # Loss under identical logits
    loss_identical = compute_ld_loss(t_logits, t_logits, temperature=args.temperature, reduction="mean")
    # Loss under student loc perturbation
    loss_biased = compute_ld_loss(s_logits, t_logits, temperature=args.temperature, reduction="mean")

    print(f"Distillation Temperature (tau) : {args.temperature}")
    print(f"LD Loss (Identical Student)     : {loss_identical:.8f} (Expected: ~0.0)")
    print(f"LD Loss (Loc-Biased Student)    : {loss_biased:.4f}")

    # 2. VLR Masking Demonstration
    anchor_pts = np.array([
        [140.0, 180.0],  # Center inside
        [85.0, 85.0],    # In VLR margin
        [20.0, 20.0],    # Far outside
    ])
    gt_boxes = np.array([[100.0, 100.0, 180.0, 260.0]])
    vlr_mask = compute_vlr_mask(anchor_pts, gt_boxes, margin_ratio=0.25)
    print(f"VLR Anchors Evaluated           : {len(anchor_pts)} (Masked Active: {np.sum(vlr_mask)})")

    # 3. Connect to TIDE Error Diagnostic
    s_batches = student_loc_biased.predict(mock_samples)
    gts = [
        GroundTruth(f"frame_{i:03d}", (100.0 + i * 20, 100.0 + i * 15, 180.0 + i * 20, 260.0 + i * 15))
        for i in range(5)
    ]
    detections = [
        Detection(batch.image_id or f"frame_{idx:03d}", tuple(batch.boxes_xyxy[0]), score=float(batch.scores[0]))
        for idx, batch in enumerate(s_batches)
    ]

    report = run_tide_diagnostics(detections, gts, fg_threshold=0.50, bg_threshold=0.10)
    print("\nTIDE Diagnostic on Student Predictions:")
    print(f"  Dominant Error Category       : {report.dominant_error}")
    print(f"  Loc Error Share               : {report.error_shares['loc']:.1%}")
    print(f"  Self-Repair Trigger Active    : {report.loc_trigger_active}")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
