"""Framework-agnostic model interfaces and deterministic mock detector (Checkpoint 4).

Defines typed array-based contracts for bounding-box predictions, teacher distillation outputs,
and detector adapters suitable for both offline testing and real detector wrappers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionBatch:
    """Standardized batch of predicted detections for a single image."""

    boxes_xyxy: np.ndarray  # Shape: (N, 4), dtype: float64
    scores: np.ndarray      # Shape: (N,), dtype: float64
    labels: np.ndarray      # Shape: (N,), dtype: int64 (0 for Person)
    image_id: str | None = None
    image_size: tuple[int, int] | None = None  # (width, height)

    def __post_init__(self) -> None:
        if self.boxes_xyxy.ndim != 2 or (self.boxes_xyxy.shape[0] > 0 and self.boxes_xyxy.shape[1] != 4):
            raise ValueError(f"boxes_xyxy must have shape (N, 4), got {self.boxes_xyxy.shape}")
        if self.scores.ndim != 1 or self.scores.shape[0] != self.boxes_xyxy.shape[0]:
            raise ValueError(
                f"scores must have shape (N,), matching boxes count {self.boxes_xyxy.shape[0]}, got {self.scores.shape}"
            )
        if self.labels.ndim != 1 or self.labels.shape[0] != self.boxes_xyxy.shape[0]:
            raise ValueError(
                f"labels must have shape (N,), matching boxes count {self.boxes_xyxy.shape[0]}, got {self.labels.shape}"
            )


@dataclass(frozen=True)
class TeacherDistillationOutputs:
    """Standardized distillation payload from a teacher detector for Localization Distillation."""

    boxes_xyxy: np.ndarray                         # Shape: (N, 4)
    scores: np.ndarray                             # Shape: (N,)
    edge_logits: np.ndarray                        # Shape: (N, 4, num_bins), typically 17 bins per edge (l, t, r, b)
    feature_maps: dict[str, np.ndarray] | None = None
    image_id: str | None = None

    def __post_init__(self) -> None:
        if self.edge_logits.ndim != 3 or (self.edge_logits.shape[0] > 0 and self.edge_logits.shape[1] != 4):
            raise ValueError(f"edge_logits must have shape (N, 4, num_bins), got {self.edge_logits.shape}")
        if self.boxes_xyxy.shape[0] != self.edge_logits.shape[0]:
            raise ValueError(
                f"boxes_xyxy count {self.boxes_xyxy.shape[0]} must match edge_logits count {self.edge_logits.shape[0]}"
            )


@runtime_checkable
class BaseDetectorAdapter(Protocol):
    """Protocol for detectors capable of generating standard predictions."""

    def predict(self, images: Sequence[Any]) -> list[PredictionBatch]:
        """Runs inference on a sequence of images or SampleRecord instances."""
        ...


@runtime_checkable
class TeacherAdapter(BaseDetectorAdapter, Protocol):
    """Protocol for teacher detectors capable of exporting distillation representations."""

    def extract_distillation_outputs(self, images: Sequence[Any]) -> list[TeacherDistillationOutputs]:
        """Extracts dense box edge distributions and optional feature maps."""
        ...


@runtime_checkable
class StudentAdapter(BaseDetectorAdapter, Protocol):
    """Protocol for student detectors receiving distillation supervision."""

    def forward_train(
        self,
        images: Sequence[Any],
        targets: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Executes student forward pass returning task predictions and student edge logits."""
        ...


class DeterministicMockDetector:
    """Lightweight, deterministic mock detector for fast offline verification without heavy ML frameworks."""

    def __init__(
        self,
        seed: int = 42,
        num_bins: int = 17,
        perturbation_mode: str = "none",
    ) -> None:
        """Initializes mock detector.

        Args:
            seed: Random seed for reproducible generation.
            num_bins: Number of discrete distribution bins per box edge (default: 17).
            perturbation_mode: 'none' (accurate detections), 'loc_biased' (shifted boxes/distributions),
                               or 'noisy'.
        """
        self.seed = seed
        self.num_bins = num_bins
        self.perturbation_mode = perturbation_mode
        self._rng = np.random.RandomState(seed)

    def _generate_edge_logits_for_box(
        self,
        box: tuple[float, float, float, float],
        shift_bins: float = 0.0,
    ) -> np.ndarray:
        """Constructs synthetic discrete probability distribution logits (4, num_bins) centered on target offsets."""
        logits = np.zeros((4, self.num_bins), dtype=np.float64)
        bin_indices = np.arange(self.num_bins, dtype=np.float64)

        center_bin = float(self.num_bins // 2) + shift_bins
        center_bin = float(np.clip(center_bin, 1.0, float(self.num_bins - 2)))

        for edge_idx in range(4):
            # Form a peaked Gaussian-like distribution over bins
            dist = np.exp(-0.5 * ((bin_indices - center_bin) / 1.5) ** 2)
            logits[edge_idx] = np.log(dist + 1e-6)

        return logits

    def predict(self, images: Sequence[Any]) -> list[PredictionBatch]:
        """Generates deterministic prediction batches for given input samples."""
        batches: list[PredictionBatch] = []

        for sample in images:
            img_id = getattr(sample, "stem", "mock_image")
            boxes_source = getattr(sample, "boxes", None)

            if boxes_source is not None and len(boxes_source) > 0:
                raw_boxes = [b.to_xyxy() if hasattr(b, "to_xyxy") else b for b in boxes_source]
            else:
                # Default synthetic boxes if input has no boxes
                raw_boxes = [(100.0, 100.0, 180.0, 240.0)]

            pred_boxes_list: list[list[float]] = []
            scores_list: list[float] = []

            for b in raw_boxes:
                x1, y1, x2, y2 = b
                w = x2 - x1
                h = y2 - y1

                if self.perturbation_mode == "loc_biased":
                    # Shift box by 50% width to trigger localization error (IoU ~ 0.33)
                    shift = 0.50 * w
                    pred_boxes_list.append([x1 + shift, y1, x2 + shift, y2])
                    scores_list.append(0.85)
                elif self.perturbation_mode == "noisy":
                    shift_x = float(self._rng.uniform(-0.1, 0.1) * w)
                    shift_y = float(self._rng.uniform(-0.1, 0.1) * h)
                    pred_boxes_list.append([x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y])
                    scores_list.append(float(self._rng.uniform(0.70, 0.95)))
                else:
                    # Near-exact detection
                    pred_boxes_list.append([x1 + 0.1, y1 + 0.1, x2 + 0.1, y2 + 0.1])
                    scores_list.append(0.95)

            boxes_arr = np.array(pred_boxes_list, dtype=np.float64)
            scores_arr = np.array(scores_list, dtype=np.float64)
            labels_arr = np.zeros(len(pred_boxes_list), dtype=np.int64)

            batches.append(
                PredictionBatch(
                    boxes_xyxy=boxes_arr,
                    scores=scores_arr,
                    labels=labels_arr,
                    image_id=img_id,
                )
            )

        return batches

    def extract_distillation_outputs(self, images: Sequence[Any]) -> list[TeacherDistillationOutputs]:
        """Generates distillation outputs with discrete edge logits."""
        pred_batches = self.predict(images)
        outputs: list[TeacherDistillationOutputs] = []

        shift_bins = 2.5 if self.perturbation_mode == "loc_biased" else 0.0

        for batch in pred_batches:
            n_boxes = batch.boxes_xyxy.shape[0]
            if n_boxes == 0:
                edge_logits = np.zeros((0, 4, self.num_bins), dtype=np.float64)
            else:
                edge_logits_list = [
                    self._generate_edge_logits_for_box(tuple(batch.boxes_xyxy[i]), shift_bins=shift_bins)
                    for i in range(n_boxes)
                ]
                edge_logits = np.stack(edge_logits_list, axis=0)

            outputs.append(
                TeacherDistillationOutputs(
                    boxes_xyxy=batch.boxes_xyxy,
                    scores=batch.scores,
                    edge_logits=edge_logits,
                    image_id=batch.image_id,
                )
            )

        return outputs

    def forward_train(
        self,
        images: Sequence[Any],
        targets: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Simulates student training forward pass."""
        outputs = self.extract_distillation_outputs(images)
        predictions = self.predict(images)

        concatenated_edge_logits = (
            np.concatenate([o.edge_logits for o in outputs], axis=0)
            if outputs
            else np.zeros((0, 4, self.num_bins), dtype=np.float64)
        )

        return {
            "predictions": predictions,
            "distillation_outputs": outputs,
            "edge_logits": concatenated_edge_logits,
        }
