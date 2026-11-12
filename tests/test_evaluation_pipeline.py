"""Unit and integration tests for localization-first evaluation and TIDE diagnostics (Checkpoint 4)."""

import unittest
import numpy as np

from src.evaluation.localization_metrics import (
    Detection,
    GroundTruth,
    compute_aspect_ratio_error,
    compute_center_error,
    compute_coco_map,
    compute_iou_matrix,
    compute_precision_recall_ap,
    compute_scale_stratified_ap,
)
from src.evaluation.tide_diagnostic import (
    DiagnosticReport,
    run_tide_diagnostics,
)


class TestIoUMatrix(unittest.TestCase):
    """Tests for vectorized bounding box IoU computation in xyxy format."""

    def test_exact_overlap(self):
        # 10x10 square at (0, 0, 10, 10)
        box = np.array([[0.0, 0.0, 10.0, 10.0]])
        iou = compute_iou_matrix(box, box)
        self.assertEqual(iou.shape, (1, 1))
        self.assertAlmostEqual(float(iou[0, 0]), 1.0)

    def test_disjoint_boxes(self):
        b1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        b2 = np.array([[20.0, 20.0, 30.0, 30.0]])
        iou = compute_iou_matrix(b1, b2)
        self.assertEqual(float(iou[0, 0]), 0.0)

    def test_partial_overlap(self):
        # Box 1: [0, 0, 10, 10], area = 100
        # Box 2: [5, 0, 15, 10], area = 100
        # Intersection: [5, 0, 10, 10], area = 5 * 10 = 50
        # Union: 100 + 100 - 50 = 150
        # IoU = 50 / 150 = 1/3
        b1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        b2 = np.array([[5.0, 0.0, 15.0, 10.0]])
        iou = compute_iou_matrix(b1, b2)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0 / 3.0, places=5)

    def test_empty_inputs(self):
        b1 = np.zeros((0, 4))
        b2 = np.array([[0.0, 0.0, 10.0, 10.0]])
        iou = compute_iou_matrix(b1, b2)
        self.assertEqual(iou.shape, (0, 1))

        iou2 = compute_iou_matrix(b2, b1)
        self.assertEqual(iou2.shape, (1, 0))


class TestGeometricErrors(unittest.TestCase):
    """Tests for normalized center displacement and aspect ratio errors."""

    def test_zero_center_error_on_identical_centers(self):
        gt = (100.0, 100.0, 200.0, 200.0)  # Center (150, 150)
        pred = (120.0, 120.0, 180.0, 180.0)  # Center (150, 150)
        err = compute_center_error(pred, gt)
        self.assertAlmostEqual(err, 0.0)

    def test_center_error_calculation(self):
        # GT: [0, 0, 60, 80] -> center = (30, 40), diag = sqrt(60^2 + 80^2) = 100
        # Pred: center shifted by (30, 40) -> pred center = (60, 80)
        # Displacement = sqrt(30^2 + 40^2) = 50
        # Normalized error = 50 / 100 = 0.50
        gt = (0.0, 0.0, 60.0, 80.0)
        pred = (30.0, 40.0, 90.0, 120.0)
        err = compute_center_error(pred, gt)
        self.assertAlmostEqual(err, 0.50)

    def test_aspect_ratio_error(self):
        # GT: w=50, h=100 -> ar = 0.50
        # Pred: w=75, h=100 -> ar = 0.75
        # Absolute diff = |0.75 - 0.50| = 0.25
        gt = (0.0, 0.0, 50.0, 100.0)
        pred = (0.0, 0.0, 75.0, 100.0)
        ar_err = compute_aspect_ratio_error(pred, gt)
        self.assertAlmostEqual(ar_err, 0.25)


class TestAveragePrecision(unittest.TestCase):
    """Tests for AP50, AP75, and mAP@[.50:.95] calculation."""

    def test_perfect_predictions(self):
        gts = [
            GroundTruth("img1", (10.0, 10.0, 50.0, 50.0)),
            GroundTruth("img2", (20.0, 20.0, 60.0, 60.0)),
        ]
        preds = [
            Detection("img1", (10.0, 10.0, 50.0, 50.0), score=0.99),
            Detection("img2", (20.0, 20.0, 60.0, 60.0), score=0.95),
        ]

        res50 = compute_precision_recall_ap(preds, gts, iou_threshold=0.50)
        self.assertAlmostEqual(res50["ap"], 1.0)
        self.assertEqual(res50["tp_count"], 2)
        self.assertEqual(res50["fp_count"], 0)

        coco_res = compute_coco_map(preds, gts)
        self.assertAlmostEqual(coco_res["map"], 1.0)
        self.assertAlmostEqual(coco_res["ap50"], 1.0)
        self.assertAlmostEqual(coco_res["ap75"], 1.0)

    def test_completely_missed_or_false_alarms(self):
        gts = [GroundTruth("img1", (10.0, 10.0, 50.0, 50.0))]
        # Pred far away
        preds = [Detection("img1", (200.0, 200.0, 250.0, 250.0), score=0.90)]

        res = compute_precision_recall_ap(preds, gts, iou_threshold=0.50)
        self.assertAlmostEqual(res["ap"], 0.0)
        self.assertEqual(res["tp_count"], 0)
        self.assertEqual(res["fp_count"], 1)

    def test_scale_stratified_ap(self):
        # 1 small (<1024), 1 medium (1024..9216), 1 large (>=9216)
        gts = [
            GroundTruth("img1", (0.0, 0.0, 20.0, 30.0)),    # area 600 -> small
            GroundTruth("img2", (0.0, 0.0, 50.0, 50.0)),    # area 2500 -> medium
            GroundTruth("img3", (0.0, 0.0, 100.0, 100.0)),  # area 10000 -> large_low_alt
        ]
        # Match only small and medium perfectly, miss large
        preds = [
            Detection("img1", (0.0, 0.0, 20.0, 30.0), score=0.95),
            Detection("img2", (0.0, 0.0, 50.0, 50.0), score=0.90),
        ]
        scale_res = compute_scale_stratified_ap(preds, gts, iou_threshold=0.50)
        self.assertAlmostEqual(scale_res["small"]["ap"], 1.0)
        self.assertAlmostEqual(scale_res["medium"]["ap"], 1.0)
        self.assertAlmostEqual(scale_res["large_low_alt"]["ap"], 0.0)


class TestTIDEDiagnostics(unittest.TestCase):
    """Tests for single-class TIDE error breakdown and self-repair trigger."""

    def test_exact_tide_error_breakdown(self):
        # Synthetic scenario with 1 of each error category:
        # 1. Correct TP: img1 box1 matched with high IoU
        # 2. Duplicate (Dup): img1 box2 also targeting box1
        # 3. Localization Error (Loc): img2 box with 0.1 <= IoU < 0.5
        # 4. Background False Alarm (Bkg): img3 box on empty background
        # 5. Missed GT (Miss): img4 box with no prediction
        gts = [
            GroundTruth("img1", (100.0, 100.0, 150.0, 150.0)),  # matched by pred1 & pred2
            GroundTruth("img2", (100.0, 100.0, 150.0, 150.0)),  # matched by pred3 with partial overlap
            GroundTruth("img4", (50.0, 50.0, 80.0, 80.0)),      # no prediction
        ]

        preds = [
            # Pred 1: High IoU (0.9+) on img1 -> TP
            Detection("img1", (102.0, 102.0, 150.0, 150.0), score=0.95),
            # Pred 2: High IoU on img1 already claimed -> Duplicate (Dup)
            Detection("img1", (100.0, 100.0, 148.0, 148.0), score=0.85),
            # Pred 3: Shifted box on img2 -> IoU ~ 0.25 (0.10 <= IoU < 0.50) -> Loc error
            Detection("img2", (125.0, 100.0, 175.0, 150.0), score=0.80),
            # Pred 4: False positive on img3 (no GTs) -> Bkg error
            Detection("img3", (500.0, 500.0, 550.0, 550.0), score=0.75),
        ]

        report = run_tide_diagnostics(preds, gts, fg_threshold=0.50, bg_threshold=0.10)

        self.assertIsInstance(report, DiagnosticReport)
        self.assertEqual(report.tp, 1)
        self.assertEqual(report.loc, 1)
        self.assertEqual(report.dup, 1)
        self.assertEqual(report.bkg, 1)
        self.assertEqual(report.miss, 1)
        self.assertEqual(report.total_errors, 4)  # loc + dup + bkg + miss

        # Relative error shares (each is 1/4 = 0.25)
        self.assertAlmostEqual(report.error_shares["loc"], 0.25)
        self.assertAlmostEqual(report.error_shares["dup"], 0.25)
        self.assertAlmostEqual(report.error_shares["bkg"], 0.25)
        self.assertAlmostEqual(report.error_shares["miss"], 0.25)

        # Trigger threshold test (0.25 < 0.40 -> False)
        self.assertFalse(report.loc_trigger_active)

    def test_tide_loc_trigger_activation(self):
        # Scenario where Loc represents >= 40% of total errors
        gts = [
            GroundTruth("img1", (100.0, 100.0, 150.0, 150.0)),
            GroundTruth("img2", (100.0, 100.0, 150.0, 150.0)),
        ]
        preds = [
            # 2 Loc errors (IoU in [0.1, 0.5))
            Detection("img1", (125.0, 100.0, 175.0, 150.0), score=0.90),
            Detection("img2", (125.0, 100.0, 175.0, 150.0), score=0.85),
        ]
        # Total errors = 2 Loc + 2 Miss (since gts weren't matched with IoU>=0.5) = 4 errors
        # Loc = 2/4 = 50% >= 40%
        report = run_tide_diagnostics(preds, gts, fg_threshold=0.50, bg_threshold=0.10)
        self.assertEqual(report.loc, 2)
        self.assertTrue(report.loc_trigger_active)
        self.assertEqual(report.dominant_error, "Loc")


if __name__ == "__main__":
    unittest.main()
