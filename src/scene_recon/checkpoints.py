"""Saving and restoring training state."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from scene_recon.environment import environment
from scene_recon.run_files import write_atomically


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

    The checkpoint also stores the complete config it was trained with, which
    is what ``scene_recon.provenance`` checks before the checkpoint is used
    again, and the software environment.

    Next to the checkpoint, a small JSON file records the training progress,
    so that tools can see how far a run has come without loading the weights.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": progress.step,
        "training_seconds": progress.training_seconds,
        "config": raw_config,
        "random_state": _global_random_state(),
        "environment": environment(),
    }
    write_atomically(path, lambda temporary_path: torch.save(state, temporary_path))
    write_atomically(
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
    load the weights for evaluation. When resuming, the global random states
    (PyTorch, Python, NumPy) are restored too, so randomness inside the model
    (dropout, stochastic depth) continues as if training had never stopped.

    This does not check that the checkpoint fits the config; call
    ``provenance.check_checkpoint_matches_config`` first.
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


def read_checkpoint_config(path: Path) -> dict[str, Any]:
    """The config a checkpoint was trained with, read without loading the weights into memory."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)["config"]


def _global_random_state() -> dict[str, Any]:
    """The state of every global random generator a model might draw from."""
    numpy_kind, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached_gauss = np.random.get_state()
    state: dict[str, Any] = {
        "cpu": torch.get_rng_state(),
        "python": random.getstate(),
        # Stored as plain numbers: restricted checkpoint loading accepts no NumPy arrays.
        "numpy": [numpy_kind, numpy_keys.tolist(), numpy_position, numpy_has_gauss, numpy_cached_gauss],
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_global_random_state(state: dict[str, Any] | None) -> None:
    if state is None:  # a checkpoint from before random states were saved
        return
    torch.set_rng_state(state["cpu"])
    if "python" in state:
        random.setstate(_as_tuples(state["python"]))
    if "numpy" in state:
        kind, keys, position, has_gauss, cached_gauss = state["numpy"]
        np.random.set_state((kind, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    # A run moved between machines may have a different number of GPUs, or none.
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def _as_tuples(value: Any) -> Any:
    """``random.setstate`` needs tuples where saving may have produced lists."""
    return tuple(_as_tuples(item) for item in value) if isinstance(value, (list, tuple)) else value


def progress_file(checkpoint_path: Path) -> Path:
    """Path of the JSON file that records the progress of a checkpoint."""
    return checkpoint_path.with_suffix(".json")
