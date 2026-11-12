"""VisDrone Person Dataset Processing Pipeline (Checkpoint 4).

Extracts and filters pedestrian (1) and people (2) categories from VisDrone2019-DET-train,
merges them into a unified 'Person' class (0), preserves bounding box coordinates and
attributes verbatim, transfers images via NTFS hardlinks (with copy fallback), and produces
a verified dataset summary report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisDroneBox:
    """Represents a single VisDrone bounding box annotation."""

    bbox_left: int
    bbox_top: int
    bbox_width: int
    bbox_height: int
    score: int
    category_id: int
    truncation: int
    occlusion: int

    def to_line(self) -> str:
        """Serializes the box back to standard VisDrone 8-field comma-separated format."""
        return (
            f"{self.bbox_left},{self.bbox_top},{self.bbox_width},{self.bbox_height},"
            f"{self.score},{self.category_id},{self.truncation},{self.occlusion}"
        )


def parse_annotation_line(line: str) -> VisDroneBox | None:
    """Parses a single line into a VisDroneBox.

    Lines with != 8 fields (such as malformed lines with trailing commas) or non-integer
    fields are discarded per strict processing specification.
    """
    cleaned = line.strip()
    if not cleaned:
        return None

    parts = cleaned.split(",")
    if len(parts) != 8:
        return None

    try:
        return VisDroneBox(
            bbox_left=int(parts[0].strip()),
            bbox_top=int(parts[1].strip()),
            bbox_width=int(parts[2].strip()),
            bbox_height=int(parts[3].strip()),
            score=int(parts[4].strip()),
            category_id=int(parts[5].strip()),
            truncation=int(parts[6].strip()),
            occlusion=int(parts[7].strip()),
        )
    except ValueError:
        return None


def filter_and_remap_box(box: VisDroneBox) -> VisDroneBox | None:
    """Filters category and remaps pedestrian/people to unified Person (0).

    Invariants:
    - Only categories 1 (pedestrian) and 2 (people) are retained.
    - Both categories are mapped to category_id = 0.
    - Width and height must be strictly positive (> 0).
    - Truncation (0, 1) and occlusion (0, 1, 2) are retained without dropping.
    """
    if box.bbox_width <= 0 or box.bbox_height <= 0:
        return None

    if box.category_id not in (1, 2):
        return None

    return VisDroneBox(
        bbox_left=box.bbox_left,
        bbox_top=box.bbox_top,
        bbox_width=box.bbox_width,
        bbox_height=box.bbox_height,
        score=box.score,
        category_id=0,
        truncation=box.truncation,
        occlusion=box.occlusion,
    )


def process_annotation_file(file_path: Path) -> tuple[list[VisDroneBox], dict[str, Any]]:
    """Processes a single annotation file, returning retained boxes and statistics."""
    retained_boxes: list[VisDroneBox] = []
    pedestrian_count = 0
    people_count = 0
    malformed_count = 0
    invalid_box_count = 0
    total_raw_lines = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_raw_lines += 1
            box = parse_annotation_line(line)
            if box is None:
                if line.strip():
                    malformed_count += 1
                continue

            if box.category_id in (1, 2) and (box.bbox_width <= 0 or box.bbox_height <= 0):
                invalid_box_count += 1

            if box.category_id == 1:
                filtered = filter_and_remap_box(box)
                if filtered is not None:
                    pedestrian_count += 1
                    retained_boxes.append(filtered)
            elif box.category_id == 2:
                filtered = filter_and_remap_box(box)
                if filtered is not None:
                    people_count += 1
                    retained_boxes.append(filtered)

    stats: dict[str, Any] = {
        "file_name": file_path.name,
        "stem": file_path.stem,
        "total_raw_lines": total_raw_lines,
        "malformed_count": malformed_count,
        "invalid_box_count": invalid_box_count,
        "pedestrian_count": pedestrian_count,
        "people_count": people_count,
        "total_retained": len(retained_boxes),
        "is_empty": len(retained_boxes) == 0,
    }
    return retained_boxes, stats


def transfer_image(src_path: Path, dst_path: Path, mode: str = "hardlink") -> str:
    """Transfers an image file using NTFS hardlink or copy with automatic fallback.

    Returns the transfer mode used ('hardlink' or 'copy').
    """
    if dst_path.exists():
        try:
            # If source and destination point to same inode/stat, no action needed
            if os.path.samefile(src_path, dst_path):
                return "hardlink"
            dst_path.unlink()
        except OSError:
            pass

    if mode == "hardlink":
        try:
            os.link(src_path, dst_path)
            return "hardlink"
        except OSError as err:
            logger.debug(
                "os.link failed for %s -> %s (%s); falling back to shutil.copy2",
                src_path,
                dst_path,
                err,
            )
            shutil.copy2(src_path, dst_path)
            return "copy"
    else:
        shutil.copy2(src_path, dst_path)
        return "copy"


def process_visdrone_dataset(
    raw_dir: Path | str,
    processed_dir: Path | str,
    link_mode: str = "hardlink",
) -> dict[str, Any]:
    """Executes the complete dataset processing pipeline.

    - Creates data/processed/visdrone_person/images/ and annotations/
    - Writes transformed annotations for all samples (generating 0-byte files for images without Person)
    - Hardlinks (or copies) all corresponding images
    - Generates and writes dataset_summary.json
    """
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)

    raw_images_dir = raw_path / "images"
    raw_annotations_dir = raw_path / "annotations"

    proc_images_dir = proc_path / "images"
    proc_annotations_dir = proc_path / "annotations"

    proc_images_dir.mkdir(parents=True, exist_ok=True)
    proc_annotations_dir.mkdir(parents=True, exist_ok=True)

    # Discover files efficiently
    image_entries: dict[str, Path] = {}
    with os.scandir(raw_images_dir) as it:
        for entry in it:
            if entry.is_file():
                p = Path(entry.path)
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    image_entries[p.stem] = p

    annotation_entries: dict[str, Path] = {}
    with os.scandir(raw_annotations_dir) as it:
        for entry in it:
            if entry.is_file():
                p = Path(entry.path)
                if p.suffix.lower() == ".txt":
                    annotation_entries[p.stem] = p

    # Stems must match
    common_stems = sorted(image_entries.keys() & annotation_entries.keys())
    if len(common_stems) != len(image_entries) or len(common_stems) != len(annotation_entries):
        missing_anns = set(image_entries.keys()) - set(annotation_entries.keys())
        missing_imgs = set(annotation_entries.keys()) - set(image_entries.keys())
        msg = (
            f"Stem mismatch: {len(image_entries)} images vs {len(annotation_entries)} annotations. "
            f"Missing annotations: {len(missing_anns)}, Missing images: {len(missing_imgs)}"
        )
        logger.error(msg)
        raise ValueError(msg)

    total_pedestrian = 0
    total_people = 0
    total_retained_boxes = 0
    total_invalid_boxes = 0
    empty_annotation_files = 0
    non_empty_annotation_files = 0
    total_malformed_lines = 0
    hardlinks_created = 0
    copies_created = 0

    truncation_counts: dict[int, int] = {}
    occlusion_counts: dict[int, int] = {}

    for stem in common_stems:
        raw_ann_file = annotation_entries[stem]
        raw_img_file = image_entries[stem]

        # 1. Process annotation
        retained_boxes, stats = process_annotation_file(raw_ann_file)
        dst_ann_file = proc_annotations_dir / f"{stem}.txt"

        if retained_boxes:
            content = "\n".join(box.to_line() for box in retained_boxes) + "\n"
            dst_ann_file.write_text(content, encoding="utf-8")
            non_empty_annotation_files += 1
        else:
            # Empty file for images with zero person instances
            dst_ann_file.write_text("", encoding="utf-8")
            empty_annotation_files += 1

        total_pedestrian += stats["pedestrian_count"]
        total_people += stats["people_count"]
        total_retained_boxes += stats["total_retained"]
        total_malformed_lines += stats["malformed_count"]
        total_invalid_boxes += stats["invalid_box_count"]

        for box in retained_boxes:
            truncation_counts[box.truncation] = truncation_counts.get(box.truncation, 0) + 1
            occlusion_counts[box.occlusion] = occlusion_counts.get(box.occlusion, 0) + 1

        # 2. Transfer image
        dst_img_file = proc_images_dir / raw_img_file.name
        mode_used = transfer_image(raw_img_file, dst_img_file, mode=link_mode)
        if mode_used == "hardlink":
            hardlinks_created += 1
        else:
            copies_created += 1

    summary: dict[str, Any] = {
        "pipeline_checkpoint": "CP4",
        "dataset_name": "visdrone_person_train",
        "source_raw_dir": str(raw_path.resolve()),
        "target_processed_dir": str(proc_path.resolve()),
        "total_images": len(common_stems),
        "total_annotation_files": len(common_stems),
        "total_retained_boxes": total_retained_boxes,
        "pedestrian_count": total_pedestrian,
        "people_count": total_people,
        "empty_annotation_files": empty_annotation_files,
        "non_empty_annotation_files": non_empty_annotation_files,
        "total_malformed_lines_skipped": total_malformed_lines,
        "total_invalid_boxes_detected": total_invalid_boxes,
        "image_transfers": {
            "hardlinks": hardlinks_created,
            "copies": copies_created,
        },
        "target_classes": {
            "0": "Person (merged pedestrian=1 and people=2)"
        },
        "truncation_distribution": {str(k): v for k, v in sorted(truncation_counts.items())},
        "occlusion_distribution": {str(k): v for k, v in sorted(occlusion_counts.items())},
    }

    summary_file = proc_path / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Processing complete: %d images, %d annotations, %d boxes (%d pedestrian, %d people)",
        len(common_stems),
        len(common_stems),
        total_retained_boxes,
        total_pedestrian,
        total_people,
    )
    return summary


def verify_dataset_invariants(
    processed_dir: Path | str,
    expected_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Performs an independent verification pass over the processed dataset directory."""
    proc_path = Path(processed_dir)
    images_dir = proc_path / "images"
    annotations_dir = proc_path / "annotations"
    summary_file = proc_path / "dataset_summary.json"

    errors: list[str] = []

    if not images_dir.exists():
        errors.append(f"Images directory missing: {images_dir}")
        return {"passed": False, "errors": errors, "metrics": {}}

    if not annotations_dir.exists():
        errors.append(f"Annotations directory missing: {annotations_dir}")
        return {"passed": False, "errors": errors, "metrics": {}}

    image_files: list[str] = []
    img_stems: set[str] = set()
    with os.scandir(images_dir) as it:
        for entry in it:
            if entry.is_file():
                stem, ext = os.path.splitext(entry.name)
                if ext.lower() in (".jpg", ".jpeg", ".png"):
                    image_files.append(entry.name)
                    img_stems.add(stem)

    annotation_entries: list[os.DirEntry[str]] = []
    ann_stems: set[str] = set()
    with os.scandir(annotations_dir) as it:
        for entry in it:
            if entry.is_file():
                stem, ext = os.path.splitext(entry.name)
                if ext.lower() == ".txt":
                    annotation_entries.append(entry)
                    ann_stems.add(stem)

    if img_stems != ann_stems:
        errors.append(
            f"1-to-1 stem mismatch: {len(img_stems)} image stems vs {len(ann_stems)} annotation stems. "
            f"Diff: {len(img_stems ^ ann_stems)}"
        )

    total_boxes = 0
    invalid_box_count = 0
    non_zero_classes: set[int] = set()
    empty_annotation_count = 0

    for entry in annotation_entries:
        line_count = 0
        with open(entry.path, "r", encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f, start=1):
                cleaned = line.strip()
                if not cleaned:
                    continue
                line_count += 1
                parts = [p.strip() for p in cleaned.split(",")]
                if len(parts) != 8:
                    errors.append(f"Malformed line {entry.name}:{line_idx} - {line}")
                    continue

                try:
                    w = int(parts[2])
                    h = int(parts[3])
                    cat = int(parts[5])

                    total_boxes += 1
                    if cat != 0:
                        non_zero_classes.add(cat)
                    if w <= 0 or h <= 0:
                        invalid_box_count += 1
                        errors.append(f"Invalid dimensions (w={w}, h={h}) at {entry.name}:{line_idx}")
                except ValueError as err:
                    errors.append(f"Non-integer value at {entry.name}:{line_idx}: {err}")

        if line_count == 0:
            empty_annotation_count += 1

    if non_zero_classes:
        errors.append(f"Found non-zero classes in output annotations: {non_zero_classes}")

    if invalid_box_count > 0:
        errors.append(f"Found {invalid_box_count} invalid bounding boxes (w<=0 or h<=0)")

    metrics = {
        "images_count": len(image_files),
        "annotations_count": len(annotation_entries),
        "retained_boxes": total_boxes,
        "empty_annotation_files": empty_annotation_count,
        "non_empty_annotation_files": len(annotation_entries) - empty_annotation_count,
        "invalid_boxes": invalid_box_count,
        "non_zero_classes": list(non_zero_classes),
        "summary_file_exists": summary_file.exists(),
    }

    if expected_metrics:
        for key, expected_val in expected_metrics.items():
            actual_val = metrics.get(key)
            if actual_val != expected_val:
                errors.append(f"Invariant check failed for '{key}': expected {expected_val}, got {actual_val}")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "metrics": metrics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for dataset processing and verification."""
    parser = argparse.ArgumentParser(
        description="Process VisDrone2019-DET-train into single-class Person dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/visdrone/extracted/VisDrone2019-DET-train"),
        help="Path to raw VisDrone dataset directory",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/visdrone_person"),
        help="Target directory for processed dataset",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="Transfer mode for images (default: hardlink with copy fallback)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification pass on existing processed directory",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    expected_cp4_invariants = {
        "images_count": 6471,
        "annotations_count": 6471,
        "retained_boxes": 106394,
        "invalid_boxes": 0,
    }

    if not args.verify_only:
        logger.info("Starting dataset processing: %s -> %s", args.raw_dir, args.processed_dir)
        summary = process_visdrone_dataset(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            link_mode=args.link_mode,
        )
        logger.info("Summary generated: %s", summary)

    logger.info("Running independent invariant verification...")
    result = verify_dataset_invariants(
        processed_dir=args.processed_dir,
        expected_metrics=expected_cp4_invariants if not args.verify_only else None,
    )

    if result["passed"]:
        logger.info("All verification invariants PASSED: %s", result["metrics"])
        return 0
    else:
        logger.error("Verification FAILED with errors:")
        for err in result["errors"]:
            logger.error("  - %s", err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
