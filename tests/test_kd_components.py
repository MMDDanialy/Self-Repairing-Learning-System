"""Unit and integration tests for Model Contracts, Localization Distillation (LD) Loss,
and Laptop Smoke-Test Subset (Checkpoint 4).
"""

import json
from pathlib import Path
import unittest
import numpy as np

from src.data.dataset_loader import BoundingBox, SampleRecord
from src.distillation.ld_loss import (
    compute_ld_loss,
    compute_vlr_mask,
    log_softmax,
    softmax,
)
from src.evaluation.localization_metrics import Detection, GroundTruth
from src.evaluation.tide_diagnostic import run_tide_diagnostics
from src.models.base_detector import (
    BaseDetectorAdapter,
    DeterministicMockDetector,
    PredictionBatch,
    StudentAdapter,
    TeacherAdapter,
    TeacherDistillationOutputs,
)
from src.data.smoke_subset import (
    SmokeSubsetManifest,
    build_smoke_subset,
    classify_sample_scale_stratum,
)


class TestLocalizationDistillationLoss(unittest.TestCase):
    """Tests for numerically stable Localization Distillation (LD) loss."""

    def setUp(self):
        self.num_bins = 17
        self.num_boxes = 10
        rng = np.random.RandomState(42)
        # Random teacher logits of shape (N, 4, 17)
        self.teacher_logits = rng.randn(self.num_boxes, 4, self.num_bins).astype(np.float64)

    def test_ld_loss_zero_when_identical(self):
        """When student and teacher edge logits are identical, LD loss must be ~0.0."""
        student_logits = np.copy(self.teacher_logits)
        loss = compute_ld_loss(student_logits, self.teacher_logits, temperature=10.0, reduction="mean")
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_ld_loss_monotonicity(self):
        """Loss should increase monotonically as student distribution diverges further from teacher."""
        rng = np.random.RandomState(123)
        direction = rng.randn(*self.teacher_logits.shape)
        direction /= np.linalg.norm(direction)

        losses = []
        perturbations = [0.0, 1.0, 5.0, 15.0, 30.0]
        for scale in perturbations:
            student_logits = self.teacher_logits + scale * direction
            loss = compute_ld_loss(student_logits, self.teacher_logits, temperature=10.0, reduction="mean")
            losses.append(float(loss))

        # Check strict monotonicity
        for i in range(len(losses) - 1):
            self.assertLess(losses[i], losses[i + 1])

    def test_ld_loss_numerical_stability_extreme_logits(self):
        """Extreme logit values (+-50.0) must not cause NaN or Inf values."""
        extreme_teacher = np.array([[[50.0] + [-50.0] * 16] * 4], dtype=np.float64)
        extreme_student = np.array([[[-50.0] * 16 + [50.0]] * 4], dtype=np.float64)

        loss = compute_ld_loss(extreme_student, extreme_teacher, temperature=10.0, reduction="mean")
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))
        self.assertGreater(float(loss), 0.0)

    def test_ld_loss_reductions(self):
        """Verify 'mean', 'sum', and 'none' reduction modes."""
        rng = np.random.RandomState(99)
        student_logits = self.teacher_logits + rng.randn(*self.teacher_logits.shape)

        loss_none = compute_ld_loss(student_logits, self.teacher_logits, reduction="none")
        self.assertEqual(loss_none.shape, (self.num_boxes,))

        loss_sum = compute_ld_loss(student_logits, self.teacher_logits, reduction="sum")
        self.assertAlmostEqual(float(loss_sum), float(np.sum(loss_none)), places=5)

        loss_mean = compute_ld_loss(student_logits, self.teacher_logits, reduction="mean")
        self.assertAlmostEqual(float(loss_mean), float(np.mean(loss_none)), places=5)

    def test_vlr_weight_masking(self):
        """Anchors with zero weight must not contribute to the loss."""
        rng = np.random.RandomState(77)
        student_logits = self.teacher_logits + rng.randn(*self.teacher_logits.shape)

        # Set weights: first 5 boxes have weight 1.0, last 5 have weight 0.0
        weights = np.array([1.0] * 5 + [0.0] * 5)
        loss_weighted = compute_ld_loss(student_logits, self.teacher_logits, box_weights=weights, reduction="mean")

        loss_first_5 = compute_ld_loss(
            student_logits[:5], self.teacher_logits[:5], box_weights=None, reduction="mean"
        )
        self.assertAlmostEqual(float(loss_weighted), float(loss_first_5), places=5)


class TestVLRMask(unittest.TestCase):
    """Tests for Valuable Localization Region (VLR) calculation."""

    def test_vlr_expansion(self):
        # Target box: [100, 100, 200, 200] -> w=100, h=100
        # Margin ratio = 0.25 -> expand by 25px on all sides: [75, 75, 225, 225]
        target_boxes = np.array([[100.0, 100.0, 200.0, 200.0]])
        anchor_points = np.array([
            [150.0, 150.0],  # Center: Inside
            [80.0, 80.0],    # In expanded VLR margin: Inside
            [70.0, 70.0],    # Outside expanded margin: Outside
            [220.0, 220.0],  # In expanded VLR margin: Inside
            [230.0, 230.0],  # Outside: Outside
        ])

        mask = compute_vlr_mask(anchor_points, target_boxes, margin_ratio=0.25)
        self.assertEqual(mask.shape, (5,))
        np.testing.assert_array_equal(mask, [True, True, False, True, False])

    def test_vlr_empty_inputs(self):
        pts = np.zeros((0, 2))
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        mask = compute_vlr_mask(pts, boxes)
        self.assertEqual(len(mask), 0)

        pts2 = np.array([[5.0, 5.0]])
        boxes2 = np.zeros((0, 4))
        mask2 = compute_vlr_mask(pts2, boxes2)
        self.assertEqual(len(mask2), 1)
        self.assertFalse(mask2[0])


class TestModelContractsAndMock(unittest.TestCase):
    """Tests for BaseDetector contracts and DeterministicMockDetector implementation."""

    def test_mock_detector_conformance(self):
        detector = DeterministicMockDetector(seed=42)
        self.assertIsInstance(detector, BaseDetectorAdapter)
        self.assertIsInstance(detector, TeacherAdapter)
        self.assertIsInstance(detector, StudentAdapter)

    def test_mock_detector_prediction_batch(self):
        detector = DeterministicMockDetector(seed=42)
        sample = SampleRecord(
            stem="sample_001",
            image_path=Path("dummy.jpg"),
            annotation_path=Path("dummy.txt"),
            boxes=(
                BoundingBox(10.0, 10.0, 50.0, 60.0),
                BoundingBox(100.0, 120.0, 80.0, 150.0),
            ),
            image_width=1000,
            image_height=800,
        )

        batches = detector.predict([sample])
        self.assertEqual(len(batches), 1)
        b = batches[0]
        self.assertIsInstance(b, PredictionBatch)
        self.assertEqual(b.boxes_xyxy.shape, (2, 4))
        self.assertEqual(b.scores.shape, (2,))
        self.assertEqual(b.labels.shape, (2,))

    def test_mock_detector_distillation_outputs(self):
        detector = DeterministicMockDetector(seed=42)
        sample = SampleRecord(
            stem="sample_002",
            image_path=Path("dummy.jpg"),
            annotation_path=Path("dummy.txt"),
            boxes=(BoundingBox(20.0, 30.0, 40.0, 50.0),),
        )

        outputs = detector.extract_distillation_outputs([sample])
        self.assertEqual(len(outputs), 1)
        out = outputs[0]
        self.assertIsInstance(out, TeacherDistillationOutputs)
        self.assertEqual(out.edge_logits.shape, (1, 4, 17))
        self.assertEqual(out.boxes_xyxy.shape, (1, 4))


class TestSmokeSubsetBuilder(unittest.TestCase):
    """Tests for the laptop smoke-test subset generator."""

    def test_classify_scale_strata(self):
        empty_sample = SampleRecord("empty", Path("a"), Path("b"), boxes=())
        self.assertEqual(classify_sample_scale_stratum(empty_sample), "empty")

        small_sample = SampleRecord(
            "small", Path("a"), Path("b"),
            boxes=(BoundingBox(0.0, 0.0, 20.0, 30.0),)  # area 600
        )
        self.assertEqual(classify_sample_scale_stratum(small_sample), "small")

        med_sample = SampleRecord(
            "med", Path("a"), Path("b"),
            boxes=(BoundingBox(0.0, 0.0, 40.0, 50.0),)  # area 2000
        )
        self.assertEqual(classify_sample_scale_stratum(med_sample), "medium")

        large_sample = SampleRecord(
            "large", Path("a"), Path("b"),
            boxes=(BoundingBox(0.0, 0.0, 100.0, 100.0),)  # area 10000
        )
        self.assertEqual(classify_sample_scale_stratum(large_sample), "large_low_alt")

    def test_smoke_subset_cardinality_and_disjointness(self):
        # Smoke check using real processed dataset if present
        config_path = Path("configs/dataset_person.json")
        if not config_path.is_file():
            self.skipTest("configs/dataset_person.json not found")

        manifest = build_smoke_subset(
            config_path=config_path,
            train_count=200,
            val_count=50,
            seed=42,
        )

        self.assertIsInstance(manifest, SmokeSubsetManifest)
        self.assertEqual(len(manifest.train_stems), 200)
        self.assertEqual(len(manifest.val_stems), 50)

        # Disjointness invariant
        train_set = set(manifest.train_stems)
        val_set = set(manifest.val_stems)
        self.assertEqual(len(train_set & val_set), 0)

        # Stratum coverage check
        for stratum in ("small", "medium", "large_low_alt", "empty"):
            self.assertGreater(manifest.strata_counts["train"].get(stratum, 0), 0)
            self.assertGreater(manifest.strata_counts["val"].get(stratum, 0), 0)


class TestEndToEndKDIntegration(unittest.TestCase):
    """Tests end-to-end integration: Teacher -> Student -> LD Loss -> TIDE diagnostic."""

    def test_mock_teacher_student_kd_pipeline(self):
        teacher = DeterministicMockDetector(seed=10, perturbation_mode="none")
        student = DeterministicMockDetector(seed=20, perturbation_mode="loc_biased")

        gts_data = [
            GroundTruth("img_01", (50.0, 50.0, 100.0, 100.0)),
            GroundTruth("img_02", (200.0, 200.0, 260.0, 280.0)),
            GroundTruth("img_03", (10.0, 10.0, 30.0, 40.0)),
        ]
        samples = [
            SampleRecord("img_01", Path("a"), Path("b"), boxes=(BoundingBox(50.0, 50.0, 50.0, 50.0),)),
            SampleRecord("img_02", Path("a"), Path("b"), boxes=(BoundingBox(200.0, 200.0, 60.0, 80.0),)),
            SampleRecord("img_03", Path("a"), Path("b"), boxes=(BoundingBox(10.0, 10.0, 20.0, 30.0),)),
        ]

        t_out = teacher.extract_distillation_outputs(samples)
        s_out = student.extract_distillation_outputs(samples)
        s_preds = student.predict(samples)

        # Compute LD loss across batches
        t_logits = np.concatenate([o.edge_logits for o in t_out], axis=0)
        s_logits = np.concatenate([o.edge_logits for o in s_out], axis=0)
        loss = compute_ld_loss(s_logits, t_logits, temperature=10.0, reduction="mean")
        self.assertGreater(float(loss), 0.0)

        # Convert student predictions to evaluation Detections
        eval_preds = []
        for batch in s_preds:
            for i in range(len(batch.boxes_xyxy)):
                eval_preds.append(
                    Detection(
                        image_id=batch.image_id or "unknown",
                        box=tuple(batch.boxes_xyxy[i]),
                        score=float(batch.scores[i]),
                    )
                )

        report = run_tide_diagnostics(eval_preds, gts_data, fg_threshold=0.50, bg_threshold=0.10)
        self.assertGreater(report.total_predictions, 0)
        self.assertTrue(report.loc_trigger_active or report.loc > 0)


if __name__ == "__main__":
    unittest.main()
