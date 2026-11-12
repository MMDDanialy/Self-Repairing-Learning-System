"""Unit and integration tests for VisDrone Person dataset processing pipeline (Checkpoint 4)."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Target module to be implemented
from src.data.process_visdrone_person import (
    VisDroneBox,
    parse_annotation_line,
    filter_and_remap_box,
    process_annotation_file,
    transfer_image,
    process_visdrone_dataset,
    verify_dataset_invariants,
)


class TestVisDroneBoxParsing(unittest.TestCase):
    """Tests for parsing VisDrone 8-field annotation lines."""

    def test_parse_valid_8_field_line(self):
        line = "684,8,273,116,0,1,0,0"
        box = parse_annotation_line(line)
        self.assertIsNotNone(box)
        self.assertEqual(box.bbox_left, 684)
        self.assertEqual(box.bbox_top, 8)
        self.assertEqual(box.bbox_width, 273)
        self.assertEqual(box.bbox_height, 116)
        self.assertEqual(box.score, 0)
        self.assertEqual(box.category_id, 1)
        self.assertEqual(box.truncation, 0)
        self.assertEqual(box.occlusion, 0)

    def test_parse_line_with_trailing_comma(self):
        # VisDrone raw annotations contain lines with trailing commas (len=9 when split)
        line = "718,1063,23,14,1,1,1,0,"
        box = parse_annotation_line(line)
        self.assertIsNone(box, "Lines with != 8 fields must be rejected as malformed")

    def test_parse_line_with_insufficient_fields(self):
        line = "718,1063,23,14,1,1"
        box = parse_annotation_line(line)
        self.assertIsNone(box)

    def test_parse_empty_or_whitespace_line(self):
        self.assertIsNone(parse_annotation_line(""))
        self.assertIsNone(parse_annotation_line("   \n"))

    def test_parse_non_integer_fields(self):
        line = "abc,10,20,30,1,1,0,0"
        box = parse_annotation_line(line)
        self.assertIsNone(box)


class TestFilteringAndRemapping(unittest.TestCase):
    """Tests for category filtering, remapping, and retention policies."""

    def test_retain_pedestrian_category_1(self):
        box = VisDroneBox(10, 20, 30, 40, 1, 1, 0, 0)
        filtered = filter_and_remap_box(box)
        self.assertIsNotNone(filtered)
        self.assertEqual(filtered.category_id, 0)  # Remapped to Person (0)
        self.assertEqual(filtered.bbox_left, 10)
        self.assertEqual(filtered.bbox_top, 20)
        self.assertEqual(filtered.bbox_width, 30)
        self.assertEqual(filtered.bbox_height, 40)
        self.assertEqual(filtered.score, 1)
        self.assertEqual(filtered.truncation, 0)
        self.assertEqual(filtered.occlusion, 0)

    def test_retain_people_category_2(self):
        box = VisDroneBox(15, 25, 35, 45, 1, 2, 0, 0)
        filtered = filter_and_remap_box(box)
        self.assertIsNotNone(filtered)
        self.assertEqual(filtered.category_id, 0)  # Remapped to Person (0)

    def test_discard_ignored_category_0(self):
        box = VisDroneBox(10, 20, 30, 40, 0, 0, 0, 0)
        self.assertIsNone(filter_and_remap_box(box))

    def test_discard_vehicle_categories(self):
        # 3: bicycle, 4: car, 5: van, 6: truck, 7: tricycle, 8: awning-tricycle, 9: bus, 10: motor, 11: others
        for cat in range(3, 12):
            box = VisDroneBox(10, 20, 30, 40, 1, cat, 0, 0)
            self.assertIsNone(filter_and_remap_box(box), f"Category {cat} should be discarded")

    def test_retain_truncated_person(self):
        # Truncation = 1 must NOT be discarded
        box = VisDroneBox(10, 20, 30, 40, 1, 1, 1, 0)
        filtered = filter_and_remap_box(box)
        self.assertIsNotNone(filtered)
        self.assertEqual(filtered.truncation, 1)

    def test_retain_occluded_person(self):
        # Occlusion = 1 (heavy) and 2 (no/slight) must NOT be discarded
        for occ in (1, 2):
            box = VisDroneBox(10, 20, 30, 40, 1, 1, 0, occ)
            filtered = filter_and_remap_box(box)
            self.assertIsNotNone(filtered)
            self.assertEqual(filtered.occlusion, occ)

    def test_reject_invalid_dimensions(self):
        # Width <= 0 or height <= 0 must be rejected
        box_zero_w = VisDroneBox(10, 20, 0, 40, 1, 1, 0, 0)
        box_zero_h = VisDroneBox(10, 20, 30, 0, 1, 1, 0, 0)
        box_neg_w = VisDroneBox(10, 20, -5, 40, 1, 1, 0, 0)
        self.assertIsNone(filter_and_remap_box(box_zero_w))
        self.assertIsNone(filter_and_remap_box(box_zero_h))
        self.assertIsNone(filter_and_remap_box(box_neg_w))


class TestAnnotationFileProcessing(unittest.TestCase):
    """Tests for single file processing, output generation, and stats."""

    def test_process_mixed_annotation_file(self):
        lines = [
            "684,8,273,116,0,0,0,0\n",      # Cat 0: ignored
            "708,471,74,33,1,4,0,1\n",       # Cat 4: car (discarded)
            "46,391,14,26,1,2,0,0\n",        # Cat 2: people (retain)
            "100,200,30,60,1,1,1,2\n",       # Cat 1: pedestrian (retain, truncated=1, occluded=2)
            "718,1063,23,14,1,1,1,0,\n",     # Malformed trailing comma (discarded)
        ]
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tf:
            tf.writelines(lines)
            tf_path = Path(tf.name)

        try:
            retained, stats = process_annotation_file(tf_path)
            self.assertEqual(len(retained), 2)
            self.assertEqual(stats["pedestrian_count"], 1)
            self.assertEqual(stats["people_count"], 1)
            self.assertEqual(stats["total_retained"], 2)
            self.assertEqual(stats["malformed_count"], 1)

            # Check format of retained boxes
            rendered = [box.to_line() for box in retained]
            self.assertEqual(rendered[0], "46,391,14,26,1,0,0,0")
            self.assertEqual(rendered[1], "100,200,30,60,1,0,1,2")
        finally:
            tf_path.unlink()

    def test_process_empty_or_non_person_file(self):
        lines = [
            "708,471,74,33,1,4,0,1\n",  # Cat 4: car
            "594,399,64,51,1,5,0,0\n",  # Cat 5: van
        ]
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tf:
            tf.writelines(lines)
            tf_path = Path(tf.name)

        try:
            retained, stats = process_annotation_file(tf_path)
            self.assertEqual(len(retained), 0)
            self.assertEqual(stats["total_retained"], 0)
            self.assertEqual(stats["is_empty"], True)
        finally:
            tf_path.unlink()


class TestImageTransfer(unittest.TestCase):
    """Tests for image transfer with hardlink and fallback to copy2."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.src_img = self.tmp_dir / "sample.jpg"
        self.src_img.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 10)
        self.dst_img = self.tmp_dir / "output" / "sample.jpg"
        self.dst_img.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_transfer_image_hardlink_success(self):
        mode_used = transfer_image(self.src_img, self.dst_img, mode="hardlink")
        self.assertTrue(self.dst_img.exists())
        self.assertEqual(self.dst_img.stat().st_size, self.src_img.stat().st_size)
        self.assertIn(mode_used, ("hardlink", "copy"))

    def test_transfer_image_fallback_to_copy_on_os_error(self):
        with patch("os.link", side_effect=OSError("Cross-device link not permitted")):
            mode_used = transfer_image(self.src_img, self.dst_img, mode="hardlink")
            self.assertEqual(mode_used, "copy")
            self.assertTrue(self.dst_img.exists())
            self.assertEqual(self.dst_img.read_bytes(), self.src_img.read_bytes())


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end integration test of dataset processing and verification."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.raw_dir = self.test_dir / "raw"
        self.raw_img_dir = self.raw_dir / "images"
        self.raw_ann_dir = self.raw_dir / "annotations"
        self.processed_dir = self.test_dir / "processed"

        self.raw_img_dir.mkdir(parents=True)
        self.raw_ann_dir.mkdir(parents=True)

        # Create 3 synthetic samples:
        # 1: 1 pedestrian + 1 car
        (self.raw_img_dir / "img1.jpg").write_bytes(b"\x00" * 100)
        (self.raw_ann_dir / "img1.txt").write_text(
            "10,20,30,40,1,1,0,0\n"  # pedestrian
            "50,60,70,80,1,4,0,0\n"  # car
        )

        # 2: 2 people
        (self.raw_img_dir / "img2.jpg").write_bytes(b"\x00" * 100)
        (self.raw_ann_dir / "img2.txt").write_text(
            "15,25,35,45,1,2,0,1\n"  # people
            "16,26,36,46,1,2,1,0\n"  # people
        )

        # 3: Only car (zero person boxes -> should generate empty annotation file)
        (self.raw_img_dir / "img3.jpg").write_bytes(b"\x00" * 100)
        (self.raw_ann_dir / "img3.txt").write_text(
            "100,100,50,50,1,4,0,0\n"
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pipeline_execution_and_invariants(self):
        summary = process_visdrone_dataset(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            link_mode="hardlink",
        )

        self.assertEqual(summary["total_images"], 3)
        self.assertEqual(summary["total_annotation_files"], 3)
        self.assertEqual(summary["total_retained_boxes"], 3)
        self.assertEqual(summary["pedestrian_count"], 1)
        self.assertEqual(summary["people_count"], 2)
        self.assertEqual(summary["empty_annotation_files"], 1)
        self.assertEqual(summary["non_empty_annotation_files"], 2)

        # Verify files on disk
        proc_imgs = list((self.processed_dir / "images").glob("*.jpg"))
        proc_anns = list((self.processed_dir / "annotations").glob("*.txt"))
        self.assertEqual(len(proc_imgs), 3)
        self.assertEqual(len(proc_anns), 3)

        # Verify empty annotation file is 0 bytes
        img3_ann = self.processed_dir / "annotations" / "img3.txt"
        self.assertTrue(img3_ann.exists())
        self.assertEqual(img3_ann.stat().st_size, 0)

        # Verify summary json exists and matches
        summary_path = self.processed_dir / "dataset_summary.json"
        self.assertTrue(summary_path.exists())
        with open(summary_path, "r") as f:
            disk_summary = json.load(f)
        self.assertEqual(disk_summary["total_retained_boxes"], 3)

        # Run verification function
        verified = verify_dataset_invariants(
            processed_dir=self.processed_dir,
            expected_metrics={
                "images_count": 3,
                "annotations_count": 3,
                "retained_boxes": 3,
            }
        )
        self.assertTrue(verified["passed"])


if __name__ == "__main__":
    unittest.main()
