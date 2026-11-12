"""Laptop Smoke-Test Subset Builder for Person Detection and Distillation (Checkpoint 4).

Samples a deterministic, stratified slice (200 train images, 50 val images) from the full
VisDrone Person dataset ensuring balanced representation of UAV altitude scale regimes
(large_low_alt, medium, small, and empty background frames) without duplicating images on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from src.data.dataset_loader import (
    DatasetSplit,
    SampleRecord,
    load_visdrone_person_dataset,
)

logger = logging.getLogger(__name__)

SCALE_STRATA = ("empty", "small", "medium", "large_low_alt")


def classify_sample_scale_stratum(sample: SampleRecord) -> str:
    """Classifies an image sample into one of four scale strata based on bounding boxes.

    Strata hierarchy:
    - 'empty': 0 Person bounding boxes (background image)
    - 'large_low_alt': at least one box >= 9,216 px^2 (5-10m AGL UAV target envelope)
    - 'medium': no large boxes, but at least one box in [1,024, 9,216) px^2
    - 'small': all boxes < 1,024 px^2
    """
    if sample.is_empty:
        return "empty"

    areas = [b.area() for b in sample.boxes]
    max_area = max(areas)

    if max_area >= 9216.0:
        return "large_low_alt"
    elif max_area >= 1024.0:
        return "medium"
    else:
        return "small"


@dataclass(frozen=True)
class SmokeSubsetManifest:
    """Immutable manifest recording sample stems and stratum statistics for laptop smoke tests."""

    train_stems: list[str]
    val_stems: list[str]
    strata_counts: dict[str, dict[str, int]]
    total_boxes: dict[str, int]
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Serializes manifest to standard dictionary."""
        return asdict(self)

    def save_json(self, output_path: Path | str) -> None:
        """Saves manifest to a JSON file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, input_path: Path | str) -> SmokeSubsetManifest:
        """Loads manifest from a JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            train_stems=data["train_stems"],
            val_stems=data["val_stems"],
            strata_counts=data["strata_counts"],
            total_boxes=data["total_boxes"],
            random_seed=data.get("random_seed", 42),
        )


def _stratified_sample_split(
    samples: Sequence[SampleRecord],
    target_count: int,
    seed: int = 42,
) -> tuple[list[SampleRecord], dict[str, int]]:
    """Performs deterministic proportional stratified sampling across scale strata."""
    if target_count >= len(samples):
        # Return all samples if requested count exceeds population
        counts = {s: sum(1 for x in samples if classify_sample_scale_stratum(x) == s) for s in SCALE_STRATA}
        return list(samples), counts

    # Partition samples by stratum
    strata_map: dict[str, list[SampleRecord]] = {s: [] for s in SCALE_STRATA}
    for s in samples:
        strata_map[classify_sample_scale_stratum(s)].append(s)

    # Sort each stratum lexicographically for deterministic cross-platform reproducibility
    for s in SCALE_STRATA:
        strata_map[s].sort(key=lambda item: item.stem)

    total_pop = len(samples)
    rng = random.Random(seed)

    # Compute ideal proportional quotas
    quotas: dict[str, int] = {}
    for stratum_name in SCALE_STRATA:
        pop_count = len(strata_map[stratum_name])
        if pop_count > 0:
            # Guarantee at least 1 sample if stratum exists
            q = max(1, int(round(target_count * (pop_count / float(total_pop)))))
            quotas[stratum_name] = min(q, pop_count)
        else:
            quotas[stratum_name] = 0

    # Adjust quotas to match exact target_count
    current_total = sum(quotas.values())
    diff = target_count - current_total

    # Add or subtract from the largest available strata
    sorted_by_size = sorted(SCALE_STRATA, key=lambda s: len(strata_map[s]), reverse=True)
    for s_name in sorted_by_size:
        if diff == 0:
            break
        pop_count = len(strata_map[s_name])
        if diff > 0 and quotas[s_name] < pop_count:
            step = min(diff, pop_count - quotas[s_name])
            quotas[s_name] += step
            diff -= step
        elif diff < 0 and quotas[s_name] > 1:
            step = min(-diff, quotas[s_name] - 1)
            quotas[s_name] -= step
            diff += step

    selected_samples: list[SampleRecord] = []
    strata_counts: dict[str, int] = {}

    for stratum_name in SCALE_STRATA:
        q = quotas[stratum_name]
        stratum_list = strata_map[stratum_name]
        indices = list(range(len(stratum_list)))
        rng.shuffle(indices)
        chosen_indices = sorted(indices[:q])
        chosen = [stratum_list[i] for i in chosen_indices]
        selected_samples.extend(chosen)
        strata_counts[stratum_name] = len(chosen)

    # Final sort by stem
    selected_samples.sort(key=lambda item: item.stem)
    return selected_samples, strata_counts


def build_smoke_subset(
    config_path: Path | str = "configs/dataset_person.json",
    train_count: int = 200,
    val_count: int = 50,
    seed: int = 42,
) -> SmokeSubsetManifest:
    """Builds a deterministic stratified smoke-test subset from the full dataset splits."""
    train_split, val_split = load_visdrone_person_dataset(config_path, split=True)

    train_subset, train_counts = _stratified_sample_split(
        train_split.samples, target_count=train_count, seed=seed
    )
    val_subset, val_counts = _stratified_sample_split(
        val_split.samples, target_count=val_count, seed=seed + 1
    )

    train_boxes = sum(len(s.boxes) for s in train_subset)
    val_boxes = sum(len(s.boxes) for s in val_subset)

    return SmokeSubsetManifest(
        train_stems=[s.stem for s in train_subset],
        val_stems=[s.stem for s in val_subset],
        strata_counts={
            "train": train_counts,
            "val": val_counts,
        },
        total_boxes={
            "train": train_boxes,
            "val": val_boxes,
        },
        random_seed=seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI runner building and exporting laptop smoke-test subset manifest."""
    parser = argparse.ArgumentParser(
        description="Generate stratified laptop smoke-test subset manifest for Person dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_person.json"),
        help="Path to dataset configuration JSON (default: configs/dataset_person.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/smoke_subset_manifest.json"),
        help="Path to save output manifest JSON (default: configs/smoke_subset_manifest.json)",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=200,
        help="Number of train samples to select (default: 200)",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=50,
        help="Number of val samples to select (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"Building stratified smoke-test subset from {args.config}...")
    manifest = build_smoke_subset(
        config_path=args.config,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
    )

    manifest.save_json(args.output)
    print(f"Saved manifest to {args.output}")

    print("\n" + "=" * 65)
    print("LAPTOP SMOKE-TEST SUBSET SUMMARY")
    print("=" * 65)
    print(f"Train Samples: {len(manifest.train_stems)} (Boxes: {manifest.total_boxes['train']:,})")
    for s, c in manifest.strata_counts["train"].items():
        print(f"  - Stratum {s:<14}: {c:3d} images ({c / len(manifest.train_stems):.1%})")

    print(f"\nVal Samples  : {len(manifest.val_stems)} (Boxes: {manifest.total_boxes['val']:,})")
    for s, c in manifest.strata_counts["val"].items():
        print(f"  - Stratum {s:<14}: {c:3d} images ({c / len(manifest.val_stems):.1%})")

    # Invariants check
    train_stems_set = set(manifest.train_stems)
    val_stems_set = set(manifest.val_stems)
    disjoint = len(train_stems_set & val_stems_set) == 0

    print("\nInvariant Checks:")
    print(f"  Exact Train Count == {args.train_count} : {'PASS' if len(manifest.train_stems) == args.train_count else 'FAIL'}")
    print(f"  Exact Val Count   == {args.val_count}   : {'PASS' if len(manifest.val_stems) == args.val_count else 'FAIL'}")
    print(f"  Disjoint Sets (Zero Leakage) : {'PASS' if disjoint else 'FAIL'}")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
