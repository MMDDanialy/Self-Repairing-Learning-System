"""Framework-Agnostic Dataset Loader for VisDrone Person Dataset (Checkpoint 4).

Loads processed single-class Person dataset, performs deterministic, reproducible,
leakage-free train/val splits, provides bounding-box coordinate conversions, and
computes altitude-aware geometric and distribution statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundingBox:
    """Represents a single Person bounding box with coordinate conversion helpers."""

    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    score: float = 1.0
    category_id: int = 0
    truncation: int = 0
    occlusion: int = 0

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Converts (bbox_left, bbox_top, bbox_width, bbox_height) to (x1, y1, x2, y2)."""
        return (
            self.bbox_left,
            self.bbox_top,
            self.bbox_left + self.bbox_width,
            self.bbox_top + self.bbox_height,
        )

    def to_xywh_norm(self, img_width: int, img_height: int) -> tuple[float, float, float, float]:
        """Converts to normalized center-based coordinates (cx_norm, cy_norm, w_norm, h_norm).

        Standard format for YOLO/anchor-free detectors:
        - cx: center x divided by image width
        - cy: center y divided by image height
        - w: bbox width divided by image width
        - h: bbox height divided by image height
        """
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {img_width}x{img_height}")

        cx = (self.bbox_left + self.bbox_width / 2.0) / float(img_width)
        cy = (self.bbox_top + self.bbox_height / 2.0) / float(img_height)
        w = self.bbox_width / float(img_width)
        h = self.bbox_height / float(img_height)
        return cx, cy, w, h

    def area(self) -> float:
        """Returns the bounding box pixel area."""
        return self.bbox_width * self.bbox_height

    def aspect_ratio(self) -> float:
        """Returns aspect ratio (width / height)."""
        return self.bbox_width / self.bbox_height if self.bbox_height > 0 else 0.0


@dataclass(frozen=True)
class SampleRecord:
    """Represents an image sample and its associated Person bounding box annotations."""

    stem: str
    image_path: Path
    annotation_path: Path
    boxes: tuple[BoundingBox, ...] = ()
    image_width: int | None = None
    image_height: int | None = None

    @property
    def is_empty(self) -> bool:
        """True if sample contains zero Person bounding boxes (background image)."""
        return len(self.boxes) == 0


@dataclass(frozen=True)
class DatasetSplit:
    """Represents a named partition (e.g. train, val) of the dataset."""

    name: str
    samples: list[SampleRecord]
    statistics: dict[str, Any]


def load_dataset_config(config_path: Path | str) -> dict[str, Any]:
    """Loads and validates the dataset configuration JSON file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Validate split ratio
    val_ratio = config.get("deterministic_val_split_ratio", 0.15)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(
            f"deterministic_val_split_ratio must be strictly between 0 and 1, got {val_ratio}"
        )

    return config


def parse_processed_annotation(file_path: Path) -> list[BoundingBox]:
    """Parses a processed Person annotation file (all category 0).

    Returns an empty list for 0-byte (empty) annotation files.
    """
    boxes: list[BoundingBox] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            cleaned = line.strip()
            if not cleaned:
                continue

            parts = [p.strip() for p in cleaned.split(",")]
            if len(parts) == 8:
                try:
                    boxes.append(
                        BoundingBox(
                            bbox_left=float(parts[0]),
                            bbox_top=float(parts[1]),
                            bbox_width=float(parts[2]),
                            bbox_height=float(parts[3]),
                            score=float(parts[4]),
                            category_id=int(parts[5]),
                            truncation=int(parts[6]),
                            occlusion=int(parts[7]),
                        )
                    )
                except ValueError:
                    continue

    return boxes


def compute_split_statistics(samples: Sequence[SampleRecord]) -> dict[str, Any]:
    """Computes summary and geometric distribution statistics for a sample set."""
    total_images = len(samples)
    empty_images = sum(1 for s in samples if s.is_empty)
    non_empty_images = total_images - empty_images

    all_boxes: list[BoundingBox] = [b for s in samples for b in s.boxes]
    total_boxes = len(all_boxes)
    mean_boxes = total_boxes / total_images if total_images > 0 else 0.0

    stats: dict[str, Any] = {
        "total_images": total_images,
        "empty_images": empty_images,
        "non_empty_images": non_empty_images,
        "total_boxes": total_boxes,
        "mean_boxes_per_image": round(mean_boxes, 3),
    }

    if total_boxes == 0:
        stats.update({
            "box_width": {},
            "box_height": {},
            "box_area": {},
            "box_aspect_ratio": {},
        })
        return stats

    widths = np.array([b.bbox_width for b in all_boxes], dtype=np.float64)
    heights = np.array([b.bbox_height for b in all_boxes], dtype=np.float64)
    areas = np.array([b.area() for b in all_boxes], dtype=np.float64)
    aspect_ratios = np.array([b.aspect_ratio() for b in all_boxes], dtype=np.float64)

    def _calc_percentiles(arr: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    stats["box_width"] = _calc_percentiles(widths)
    stats["box_height"] = _calc_percentiles(heights)
    stats["box_area"] = _calc_percentiles(areas)
    stats["box_aspect_ratio"] = _calc_percentiles(aspect_ratios)

    return stats


def create_splits(
    samples: Sequence[SampleRecord],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DatasetSplit, DatasetSplit]:
    """Partitions samples deterministically into train and val splits without leakage.

    Samples are sorted by stem lexicographically before shuffling to ensure
    identical split assignments regardless of directory ordering on disk.
    """
    sorted_samples = sorted(samples, key=lambda s: s.stem)
    rng = random.Random(seed)
    shuffled_indices = list(range(len(sorted_samples)))
    rng.shuffle(shuffled_indices)

    val_size = int(round(len(sorted_samples) * val_ratio))
    val_indices = set(shuffled_indices[:val_size])

    train_samples = [sorted_samples[i] for i in range(len(sorted_samples)) if i not in val_indices]
    val_samples = [sorted_samples[i] for i in range(len(sorted_samples)) if i in val_indices]

    train_stats = compute_split_statistics(train_samples)
    val_stats = compute_split_statistics(val_samples)

    return (
        DatasetSplit("train", train_samples, train_stats),
        DatasetSplit("val", val_samples, val_stats),
    )


def load_visdrone_person_dataset(
    config_path: Path | str,
    split: bool = True,
) -> tuple[DatasetSplit, DatasetSplit] | list[SampleRecord]:
    """Loads all processed VisDrone Person samples and creates deterministic splits."""
    config = load_dataset_config(config_path)
    processed_dir = Path(config["processed_dir"])
    images_dir = processed_dir / "images"
    annotations_dir = processed_dir / "annotations"

    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Processed dataset directories missing under {processed_dir}. "
            "Ensure process_visdrone_person.py has been run."
        )

    # Discover matched image and annotation pairs
    image_paths: dict[str, Path] = {}
    with os.scandir(images_dir) as it:
        for entry in it:
            if entry.is_file():
                p = Path(entry.path)
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    image_paths[p.stem] = p

    annotation_paths: dict[str, Path] = {}
    with os.scandir(annotations_dir) as it:
        for entry in it:
            if entry.is_file():
                p = Path(entry.path)
                if p.suffix.lower() == ".txt":
                    annotation_paths[p.stem] = p

    common_stems = sorted(image_paths.keys() & annotation_paths.keys())
    include_empty = config.get("include_empty_background_images", True)

    samples: list[SampleRecord] = []
    for stem in common_stems:
        ann_path = annotation_paths[stem]
        img_path = image_paths[stem]
        boxes = parse_processed_annotation(ann_path)

        if not boxes and not include_empty:
            continue

        samples.append(
            SampleRecord(
                stem=stem,
                image_path=img_path,
                annotation_path=ann_path,
                boxes=tuple(boxes),
            )
        )

    if not split:
        return samples

    val_ratio = float(config.get("deterministic_val_split_ratio", 0.15))
    seed = int(config.get("random_seed", 42))
    return create_splits(samples, val_ratio=val_ratio, seed=seed)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI smoke runner printing train vs val statistics from processed dataset."""
    parser = argparse.ArgumentParser(
        description="Load VisDrone Person dataset and print train/val split statistics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_person.json"),
        help="Path to dataset configuration JSON (default: configs/dataset_person.json)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"Loading dataset configuration from: {args.config}")
    train_split, val_split = load_visdrone_person_dataset(args.config)

    print("\n" + "=" * 65)
    print("CP4 FRAMEWORK-AGNOSTIC DATASET LOADER — SPLIT SUMMARY")
    print("=" * 65)

    for split in (train_split, val_split):
        stats = split.statistics
        print(f"\n--- Split: {split.name.upper()} ---")
        print(f"  Total Images       : {stats['total_images']:,}")
        print(f"  Non-empty Images   : {stats['non_empty_images']:,}")
        print(f"  Empty (Background) : {stats['empty_images']:,}")
        print(f"  Total Person Boxes : {stats['total_boxes']:,}")
        print(f"  Mean Boxes/Image   : {stats['mean_boxes_per_image']:.2f}")

        if stats["total_boxes"] > 0:
            bw = stats["box_width"]
            bh = stats["box_height"]
            ba = stats["box_area"]
            bar = stats["box_aspect_ratio"]
            print("  Box Geometry Percentiles:")
            print(f"    Width  (px) : p25={bw['p25']:.1f}, median={bw['p50']:.1f}, p75={bw['p75']:.1f}, p95={bw['p95']:.1f} (min={bw['min']:.0f}, max={bw['max']:.0f})")
            print(f"    Height (px) : p25={bh['p25']:.1f}, median={bh['p50']:.1f}, p75={bh['p75']:.1f}, p95={bh['p95']:.1f} (min={bh['min']:.0f}, max={bh['max']:.0f})")
            print(f"    Area (px^2) : p25={ba['p25']:.0f}, median={ba['p50']:.0f}, p75={ba['p75']:.0f}, p95={ba['p95']:.0f}")
            print(f"    Aspect Ratio: median={bar['p50']:.2f}, mean={bar['mean']:.2f}")

    print("\n" + "=" * 65)
    print("Verification Invariants Check:")
    total_imgs = len(train_split.samples) + len(val_split.samples)
    total_boxes = train_split.statistics["total_boxes"] + val_split.statistics["total_boxes"]
    train_stems = {s.stem for s in train_split.samples}
    val_stems = {s.stem for s in val_split.samples}
    disjoint = len(train_stems & val_stems) == 0

    print(f"  Total Images: {total_imgs} (Expected: 6,471) -> {'PASS' if total_imgs == 6471 else 'FAIL'}")
    print(f"  Total Boxes : {total_boxes} (Expected: 106,394) -> {'PASS' if total_boxes == 106394 else 'FAIL'}")
    print(f"  Zero Leakage: Disjoint Stems -> {'PASS' if disjoint else 'FAIL'}")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
