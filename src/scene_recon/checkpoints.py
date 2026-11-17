"""Saving and restoring training state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

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
    """Write everything needed to resume training or to evaluate the model.

    Next to the checkpoint, a small JSON file records the training progress,
    so that tools can see how far a run has come without loading the weights.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": progress.step,
        "training_seconds": progress.training_seconds,
        "config": raw_config,
        "random_state": _global_random_state(),
    }
    _write_atomically(path, lambda temporary_path: torch.save(state, temporary_path))
    _write_atomically(
        progress_file(path), lambda temporary_path: temporary_path.write_text(json.dumps(asdict(progress)))
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> TrainingProgress:
    """Restore a model, and optionally the optimiser and scheduler, from a checkpoint.

    Pass the optimiser and scheduler to resume training; leave them out to
    load the weights for evaluation. When resuming, PyTorch's global random
    state is restored too, so randomness inside the model (dropout,
    stochastic depth) continues as if training had never stopped.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}. Train the model first, or pass --checkpoint.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        _restore_global_random_state(state.get("random_state"))
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return TrainingProgress(step=state["step"], training_seconds=state["training_seconds"])


def read_progress(checkpoint_path: Path) -> TrainingProgress | None:
    """The training progress stored with a checkpoint, or ``None`` if there is no checkpoint."""
    if not checkpoint_path.is_file():
        return None

    sidecar = progress_file(checkpoint_path)
    if sidecar.is_file():
        return TrainingProgress(**json.loads(sidecar.read_text()))

    # A checkpoint without the JSON file was not written by save_checkpoint;
    # fall back to reading the (much larger) checkpoint itself.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return TrainingProgress(step=state["step"], training_seconds=state["training_seconds"])


def _global_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_global_random_state(state: dict[str, Any] | None) -> None:
    if state is None:  # a checkpoint from before random states were saved
        return
    torch.set_rng_state(state["cpu"])
    # A run moved between machines may have a different number of GPUs, or none.
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def progress_file(checkpoint_path: Path) -> Path:
    """Path of the JSON file that records the progress of a checkpoint."""
    return checkpoint_path.with_suffix(".json")


def _write_atomically(path: Path, write: Callable[[Path], Any]) -> None:
    """Write to a temporary file, then move it into place.

    A crash while writing therefore never leaves a truncated file in place of
    a good one.
    """
    temporary_path = path.with_name(path.name + ".tmp")
    write(temporary_path)
    temporary_path.replace(path)
