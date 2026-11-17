"""Saving and restoring training state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass
class TrainingProgress:
    """How far a training run has come."""

    step: int = 0
    """Number of completed optimisation steps."""

    training_seconds: float = 0.0
    """Wall-clock time spent in optimisation steps, accumulated across resumed runs.

    Excludes validation and checkpointing. On a single GPU,
    ``training_seconds / 3600`` is the "training GPU hours" figure of the proposal.
    """


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    progress: TrainingProgress,
    raw_config: dict[str, Any],
) -> None:
    """Write everything needed to resume training or to evaluate the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": progress.step,
        "training_seconds": progress.training_seconds,
        "config": raw_config,
    }
    # Write to a temporary file first, so that a crash while saving cannot
    # leave a truncated checkpoint in place of a good one.
    temporary_path = path.with_suffix(".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> TrainingProgress:
    """Restore a model, and optionally the optimiser and scheduler, from a checkpoint.

    Pass the optimiser and scheduler to resume training; leave them out to
    load the weights for evaluation.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return TrainingProgress(step=state["step"], training_seconds=state["training_seconds"])
