"""Model interfaces, adapters, and deterministic mocks."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.base_detector import (
        BaseDetectorAdapter,
        DeterministicMockDetector,
        PredictionBatch,
        StudentAdapter,
        TeacherAdapter,
        TeacherDistillationOutputs,
    )


def __getattr__(name: str):
    if name in (
        "BaseDetectorAdapter",
        "DeterministicMockDetector",
        "PredictionBatch",
        "StudentAdapter",
        "TeacherAdapter",
        "TeacherDistillationOutputs",
    ):
        from src.models import base_detector

        return getattr(base_detector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseDetectorAdapter",
    "DeterministicMockDetector",
    "PredictionBatch",
    "StudentAdapter",
    "TeacherAdapter",
    "TeacherDistillationOutputs",
]
