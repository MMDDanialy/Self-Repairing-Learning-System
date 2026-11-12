"""TIDE-Style Single-Class Diagnostic Engine for VisDrone Person (Checkpoint 4).

Decomposes detection errors into:
- TP: Correct detection (IoU >= fg_threshold, highest-confidence match)
- Loc: Localization error (bg_threshold <= IoU < fg_threshold)
- Dup: Duplicate detection (IoU >= fg_threshold on already-claimed GT)
- Bkg: Background false positive (IoU < bg_threshold or detection on empty image)
- Miss: Missed ground truth (unmatched GT instances)

Directly drives CP5 adaptive self-repair KD intervention when Loc error share >= 40%.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from src.evaluation.localization_metrics import Detection, GroundTruth, compute_iou_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiagnosticReport:
    """Structured diagnostic summary report quantifying error counts, shares, and triggers."""

    tp: int
    loc: int
    dup: int
    bkg: int
    miss: int
    total_predictions: int
    total_gt: int
    total_errors: int
    error_shares: dict[str, float]
    dominant_error: str
    loc_trigger_active: bool
    fg_threshold: float = 0.50
    bg_threshold: float = 0.10
    loc_trigger_threshold: float = 0.40

    def to_dict(self) -> dict[str, Any]:
        """Converts report to dictionary representation for serialization."""
        return asdict(self)


def run_tide_diagnostics(
    preds: Sequence[Detection],
    gts: Sequence[GroundTruth],
    fg_threshold: float = 0.50,
    bg_threshold: float = 0.10,
    loc_trigger_threshold: float = 0.40,
) -> DiagnosticReport:
    """Performs deterministic single-class TIDE error decomposition on predictions vs GT.

    Args:
        preds: Sequence of predicted Detections.
        gts: Sequence of ground-truth annotations.
        fg_threshold: Minimum IoU to qualify as foreground True Positive (default: 0.50).
        bg_threshold: Minimum IoU to qualify as Localization error rather than Background (default: 0.10).
        loc_trigger_threshold: Ratio of Loc / total_errors to activate self-repair loop (default: 0.40).

    Returns:
        DiagnosticReport summarizing error counts, error shares, and active triggers.
    """
    # Group by image_id
    gt_by_img: dict[str, list[GroundTruth]] = {}
    for g in gts:
        gt_by_img.setdefault(g.image_id, []).append(g)

    pred_by_img: dict[str, list[Detection]] = {}
    for p in preds:
        pred_by_img.setdefault(p.image_id, []).append(p)

    all_image_ids = set(gt_by_img.keys()) | set(pred_by_img.keys())

    tp_count = 0
    loc_count = 0
    dup_count = 0
    bkg_count = 0
    miss_count = 0

    for img_id in all_image_ids:
        img_gts = gt_by_img.get(img_id, [])
        img_preds = sorted(pred_by_img.get(img_id, []), key=lambda p: p.score, reverse=True)

        # Case 1: Image has detections but no GT (pure background false alarms)
        if not img_gts:
            bkg_count += len(img_preds)
            continue

        # Case 2: Image has GT but no detections (all missed)
        if not img_preds:
            miss_count += len(img_gts)
            continue

        p_boxes = np.array([p.box for p in img_preds])
        g_boxes = np.array([g.box for g in img_gts])
        iou_mat = compute_iou_matrix(p_boxes, g_boxes)  # Shape (P, G)

        assigned_tp_gts: set[int] = set()
        covered_gts: set[int] = set()

        for p_idx in range(len(img_preds)):
            ious = iou_mat[p_idx]
            best_gt_idx = int(np.argmax(ious))
            max_iou = float(ious[best_gt_idx])

            if max_iou >= fg_threshold:
                if best_gt_idx not in assigned_tp_gts:
                    tp_count += 1
                    assigned_tp_gts.add(best_gt_idx)
                    covered_gts.add(best_gt_idx)
                else:
                    dup_count += 1
            elif max_iou >= bg_threshold:
                loc_count += 1
                covered_gts.add(best_gt_idx)
            else:
                bkg_count += 1

        unmatched_gt_count = len(img_gts) - len(covered_gts)
        miss_count += unmatched_gt_count

    total_errors = loc_count + dup_count + bkg_count + miss_count

    if total_errors > 0:
        error_shares = {
            "loc": loc_count / total_errors,
            "dup": dup_count / total_errors,
            "bkg": bkg_count / total_errors,
            "miss": miss_count / total_errors,
        }
        # Find dominant error category
        error_counts = {
            "Loc": loc_count,
            "Dup": dup_count,
            "Bkg": bkg_count,
            "Miss": miss_count,
        }
        dominant_error = max(error_counts.items(), key=lambda kv: kv[1])[0]
    else:
        error_shares = {"loc": 0.0, "dup": 0.0, "bkg": 0.0, "miss": 0.0}
        dominant_error = "None"

    loc_trigger_active = error_shares["loc"] >= loc_trigger_threshold

    return DiagnosticReport(
        tp=tp_count,
        loc=loc_count,
        dup=dup_count,
        bkg=bkg_count,
        miss=miss_count,
        total_predictions=len(preds),
        total_gt=len(gts),
        total_errors=total_errors,
        error_shares=error_shares,
        dominant_error=dominant_error,
        loc_trigger_active=loc_trigger_active,
        fg_threshold=fg_threshold,
        bg_threshold=bg_threshold,
        loc_trigger_threshold=loc_trigger_threshold,
    )


def create_synthetic_val_fixture(
    val_samples: Sequence[Any],
    loc_bias: bool = True,
) -> tuple[list[GroundTruth], list[Detection]]:
    """Generates synthetic predictions against real val split ground truth to simulate a student detector.

    Simulates realistic detector artifacts:
    - 40% TP detections with minor jitter
    - 25% Loc errors with shifted bounding boxes
    - 15% Duplicate detections
    - 10% Background false alarms
    - 20% Missed ground truth instances
    """
    gts: list[GroundTruth] = []
    preds: list[Detection] = []

    for idx, sample in enumerate(val_samples):
        img_id = sample.stem
        sample_gts: list[GroundTruth] = []
        for b in sample.boxes:
            gt_box = b.to_xyxy()
            sample_gts.append(GroundTruth(image_id=img_id, box=gt_box))
        gts.extend(sample_gts)

        if not sample_gts:
            # Empty image: simulate occasional background false alarm
            if idx % 3 == 0:
                preds.append(
                    Detection(
                        image_id=img_id,
                        box=(100.0, 100.0, 180.0, 220.0),
                        score=0.65,
                    )
                )
            continue

        for g_idx, g in enumerate(sample_gts):
            x1, y1, x2, y2 = g.box
            w = x2 - x1
            h = y2 - y1

            mode = (idx + g_idx) % 5
            if mode in (0, 1):
                # TP: slight jitter (~0.90 IoU)
                preds.append(
                    Detection(
                        image_id=img_id,
                        box=(x1 + 0.5, y1 + 0.5, x2 + 0.5, y2 + 0.5),
                        score=0.92,
                    )
                )
                if mode == 1:
                    # Duplicate detection
                    preds.append(
                        Detection(
                            image_id=img_id,
                            box=(x1 - 0.5, y1 - 0.5, x2 - 0.5, y2 - 0.5),
                            score=0.81,
                        )
                    )
            elif mode in (2, 3) if loc_bias else (2,):
                # Loc error: shifted box (IoU in [0.20, 0.45])
                shift_x = 0.55 * w
                preds.append(
                    Detection(
                        image_id=img_id,
                        box=(x1 + shift_x, y1, x2 + shift_x, y2),
                        score=0.78,
                    )
                )
            elif mode == 4:
                # Miss: do not add a prediction for this GT
                pass

        # Occasional background false alarm
        if idx % 4 == 0:
            preds.append(
                Detection(
                    image_id=img_id,
                    box=(500.0, 500.0, 550.0, 600.0),
                    score=0.60,
                )
            )

    return gts, preds


def main(argv: Sequence[str] | None = None) -> int:
    """CLI runner executing synthetic and validation TIDE diagnostic demonstration."""
    parser = argparse.ArgumentParser(
        description="Run TIDE error diagnostic demonstration for Person detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_person.json",
        help="Path to dataset configuration JSON (default: configs/dataset_person.json)",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=25,
        help="Number of val samples to draw for demo fixture (default: 25, 0 for pure synthetic)",
    )
    parser.add_argument(
        "--fg-thresh",
        type=float,
        default=0.50,
        help="Foreground IoU threshold for TP (default: 0.50)",
    )
    parser.add_argument(
        "--bg-thresh",
        type=float,
        default=0.10,
        help="Background IoU threshold for Loc (default: 0.10)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.val_samples > 0:
        from pathlib import Path
        from src.data.dataset_loader import load_visdrone_person_dataset

        config_path = Path(args.config)
        if config_path.is_file():
            print(f"Loading {args.val_samples} validation samples from: {config_path}")
            _, val_split = load_visdrone_person_dataset(config_path)
            sample_slice = val_split.samples[: args.val_samples]
            gts, preds = create_synthetic_val_fixture(sample_slice)
            fixture_desc = f"Val Split Fixture ({args.val_samples} images, {len(gts)} GTs, {len(preds)} Preds)"
        else:
            print(f"Config {config_path} not found; falling back to in-memory synthetic fixture.")
            gts, preds = None, None
            fixture_desc = "Synthetic Hardcoded Fixture"
    else:
        gts, preds = None, None
        fixture_desc = "Synthetic Hardcoded Fixture"

    if gts is None or preds is None:
        # Fallback in-memory synthetic fixture
        gts = [
            GroundTruth("frame_001", (100.0, 100.0, 150.0, 150.0)),
            GroundTruth("frame_001", (200.0, 200.0, 240.0, 280.0)),
            GroundTruth("frame_002", (50.0, 50.0, 80.0, 120.0)),
            GroundTruth("frame_002", (300.0, 300.0, 350.0, 400.0)),
            GroundTruth("frame_003", (120.0, 150.0, 180.0, 270.0)),
        ]

        preds = [
            # Frame 1: 1 TP, 1 Duplicate
            Detection("frame_001", (101.0, 101.0, 150.0, 150.0), score=0.96),
            Detection("frame_001", (99.0, 102.0, 149.0, 151.0), score=0.88),  # Dup
            # Frame 2: 1 Loc error (shifted box, IoU ~0.35), 1 Miss (frame_002 box 2 untouched)
            Detection("frame_002", (65.0, 50.0, 95.0, 120.0), score=0.82),   # Loc
            # Frame 3: 1 Bkg false alarm on empty background
            Detection("frame_003", (500.0, 500.0, 560.0, 620.0), score=0.74),  # Bkg
        ]
        fixture_desc = "Synthetic Hardcoded Fixture (3 frames, 5 GTs, 4 Preds)"

    report = run_tide_diagnostics(
        preds,
        gts,
        fg_threshold=args.fg_thresh,
        bg_threshold=args.bg_thresh,
    )

    print("\n" + "=" * 65)
    print(f"CP4 TIDE-STYLE SINGLE-CLASS DIAGNOSTIC REPORT ({fixture_desc.upper()})")
    print("=" * 65)
    print(json.dumps(report.to_dict(), indent=2))
    print("=" * 65)
    print(f"Dominant Failure Mode : {report.dominant_error}")
    print(f"Self-Repair Trigger Active: {report.loc_trigger_active} (Loc Share = {report.error_shares['loc']:.1%})")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
