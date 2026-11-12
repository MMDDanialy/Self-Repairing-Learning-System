"""Distillation losses and region masking engines."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.distillation.ld_loss import (
        compute_ld_loss,
        compute_vlr_mask,
        log_softmax,
        softmax,
    )


def __getattr__(name: str):
    if name in ("compute_ld_loss", "compute_vlr_mask", "log_softmax", "softmax"):
        from src.distillation import ld_loss

        return getattr(ld_loss, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compute_ld_loss",
    "compute_vlr_mask",
    "log_softmax",
    "softmax",
]
