"""Unit and integration tests for framework-agnostic dataset loader (Checkpoint 4)."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.data.dataset_loader import (
    BoundingBox,
    DatasetSplit,
    SampleRecord,
    compute_split_statistics,
    create_splits,
    load_dataset_config,
    load_visdrone_person_dataset,
    parse_processed_annotation,
)


class TestDatasetConfig(unittest.TestCase):
    """Tests for dataset configuration loading and validation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "test_config.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_config(self):
        payload = {
            "dataset_name": "visdrone_person",
            "processed_dir": "data/processed/visdrone_person",
            "class_names": ["Person"],
            "num_classes": 1,
            "bbox_format": "xywh_abs",
            "supported_target_formats": ["xyxy_abs", "xywh_norm"],
            "deterministic_val_split_ratio": 0.15,
            "random_seed": 42,
            "include_empty_background_images": True,
        }
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")
        cfg = load_dataset_config(self.config_path)

        self.assertEqual(cfg["dataset_name"], "visdrone_person")
        self.assertEqual(cfg["deterministic_val_split_ratio"], 0.15)
        self.assertEqual(cfg["random_seed"], 42)
        self.assertTrue(cfg["include_empty_background_images"])

    def test_missing_config_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset_config(Path(self.temp_dir.name) / "non_existent.json")

    def test_invalid_split_ratio_raises_value_error(self):
        payload = {
            "dataset_name": "visdrone_person",
            "processed_dir": "data/processed/visdrone_person",
            "class_names": ["Person"],
            "num_classes": 1,
            "deterministic_val_split_ratio": 1.5,  # Invalid
            "random_seed": 42,
        }
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_dataset_config(self.config_path)


class TestBoundingBox(unittest.TestCase):
    """Tests for BoundingBox geometric transformations and properties."""

    def setUp(self):
        # VisDrone box format: left=100, top=200, width=50, height=100, score=1, class=0, trunc=0, occ=1
        self.box = BoundingBox(
            bbox_left=100.0,
            bbox_top=200.0,
            bbox_width=50.0,
            bbox_height=100.0,
            score=1.0,
            category_id=0,
            truncation=0,
            occlusion=1,
        )

    def test_to_xyxy(self):
        x1, y1, x2, y2 = self.box.to_xyxy()
        self.assertEqual((x1, y1, x2, y2), (100.0, 200.0, 150.0, 300.0))

    def test_to_xywh_norm(self):
        img_w, img_h = 1000, 1000
        # Center x = 100 + 25 = 125 -> 0.125
        # Center y = 200 + 50 = 250 -> 0.250
        # Norm w = 50 / 1000 = 0.050
        # Norm h = 100 / 1000 = 0.100
        cx, cy, nw, nh = self.box.to_xywh_norm(img_w, img_h)
        self.assertAlmostEqual(cx, 0.125)
        self.assertAlmostEqual(cy, 0.250)
        self.assertAlmostEqual(nw, 0.050)
        self.assertAlmostEqual(nh, 0.100)

    def test_area(self):
        self.assertEqual(self.box.area(), 5000.0)

    def test_aspect_ratio(self):
        # aspect ratio = width / height = 50 / 100 = 0.5
        self.assertAlmostEqual(self.box.aspect_ratio(), 0.5)


class TestAnnotationParsing(unittest.TestCase):
    """Tests for parsing processed VisDrone Person annotation files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_empty_annotation_file(self):
        empty_file = Path(self.temp_dir.name) / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        boxes = parse_processed_annotation(empty_file)
        self.assertEqual(len(boxes), 0)

    def test_parse_multi_box_file(self):
        ann_file = Path(self.temp_dir.name) / "sample.txt"
        content = (
            "46,391,14,26,1,0,0,0\n"
            "100,200,30,60,1,0,1,2\n"
        )
        ann_file.write_text(content, encoding="utf-8")
        boxes = parse_processed_annotation(ann_file)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0].category_id, 0)
        self.assertEqual(boxes[0].bbox_width, 14.0)
        self.assertEqual(boxes[1].category_id, 0)
        self.assertEqual(boxes[1].truncation, 1)
        self.assertEqual(boxes[1].occlusion, 2)


class TestDeterministicSplits(unittest.TestCase):
    """Tests for reproducible, deterministic train/val splitting."""

    def _create_dummy_samples(self, n: int) -> list[SampleRecord]:
        samples = []
        for i in range(n):
            stem = f"{i:05d}"
            samples.append(
                SampleRecord(
                    stem=stem,
                    image_path=Path(f"/tmp/imgs/{stem}.jpg"),
                    annotation_path=Path(f"/tmp/anns/{stem}.txt"),
                    boxes=(
                        BoundingBox(10.0, 20.0, 30.0, 40.0, 1.0, 0, 0, 0),
                    ) if i % 2 == 0 else (),
                )
            )
        return samples

    def test_split_cardinality_and_disjointness(self):
        samples = self._create_dummy_samples(100)
        train_split, val_split = create_splits(samples, val_ratio=0.15, seed=42)

        self.assertEqual(len(train_split.samples), 85)
        self.assertEqual(len(val_split.samples), 15)

        train_stems = {s.stem for s in train_split.samples}
        val_stems = {s.stem for s in val_split.samples}

        # Zero leakage: empty intersection
        self.assertEqual(train_stems & val_stems, set())
        # Completeness: full union
        self.assertEqual(train_stems | val_stems, {s.stem for s in samples})

    def test_split_reproducibility(self):
        samples = self._create_dummy_samples(100)
        t1, v1 = create_splits(samples, val_ratio=0.15, seed=42)
        t2, v2 = create_splits(samples, val_ratio=0.15, seed=42)

        self.assertEqual([s.stem for s in t1.samples], [s.stem for s in t2.samples])
        self.assertEqual([s.stem for s in v1.samples], [s.stem for s in v2.samples])

    def test_different_seeds_produce_different_splits(self):
        samples = self._create_dummy_samples(100)
        t1, _ = create_splits(samples, val_ratio=0.15, seed=42)
        t2, _ = create_splits(samples, val_ratio=0.15, seed=99)

        self.assertNotEqual([s.stem for s in t1.samples], [s.stem for s in t2.samples])


class TestSplitStatistics(unittest.TestCase):
    """Tests for split statistics computation and percentiles."""

    def test_statistics_calculation(self):
        samples = [
            SampleRecord(
                stem="001",
                image_path=Path("001.jpg"),
                annotation_path=Path("001.txt"),
                boxes=(
                    BoundingBox(0.0, 0.0, 20.0, 40.0, 1.0, 0, 0, 0),  # area 800, ar 0.5
                    BoundingBox(0.0, 0.0, 40.0, 80.0, 1.0, 0, 0, 0),  # area 3200, ar 0.5
                ),
            ),
            SampleRecord(
                stem="002",
                image_path=Path("002.jpg"),
                annotation_path=Path("002.txt"),
                boxes=(),  # empty
            ),
        ]
        stats = compute_split_statistics(samples)

        self.assertEqual(stats["total_images"], 2)
        self.assertEqual(stats["empty_images"], 1)
        self.assertEqual(stats["non_empty_images"], 1)
        self.assertEqual(stats["total_boxes"], 2)
        self.assertAlmostEqual(stats["mean_boxes_per_image"], 1.0)

        # Check box percentiles
        self.assertEqual(stats["box_width"]["p50"], 30.0)
        self.assertEqual(stats["box_height"]["p50"], 60.0)
        self.assertEqual(stats["box_area"]["p50"], 2000.0)

    def test_statistics_empty_samples(self):
        stats = compute_split_statistics([])
        self.assertEqual(stats["total_images"], 0)
        self.assertEqual(stats["total_boxes"], 0)


class TestEndToEndSmoke(unittest.TestCase):
    """Smoke test running on real processed VisDrone Person dataset if present."""

    def test_load_real_processed_dataset(self):
        processed_dir = Path("data/processed/visdrone_person")
        if not (processed_dir / "images").exists() or not (processed_dir / "annotations").exists():
            self.skipTest("Processed VisDrone Person dataset not found on disk")

        # Create temporary config pointing to real dataset
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tf:
            cfg = {
                "dataset_name": "visdrone_person",
                "processed_dir": str(processed_dir),
                "class_names": ["Person"],
                "num_classes": 1,
                "bbox_format": "xywh_abs",
                "deterministic_val_split_ratio": 0.15,
                "random_seed": 42,
                "include_empty_background_images": True,
            }
            json.dump(cfg, tf)
            cfg_path = Path(tf.name)

        try:
            train_split, val_split = load_visdrone_person_dataset(cfg_path)
            self.assertEqual(len(train_split.samples) + len(val_split.samples), 6471)
            # 15% of 6471 = 970.65 -> 970 or 971
            self.assertIn(len(val_split.samples), (970, 971))
            self.assertIn(len(train_split.samples), (5500, 5501))

            total_boxes = train_split.statistics["total_boxes"] + val_split.statistics["total_boxes"]
            self.assertEqual(total_boxes, 106394)
        finally:
            cfg_path.unlink()


if __name__ == "__main__":
    unittest.main()
