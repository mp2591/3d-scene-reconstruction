"""Saving and restoring training state."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from scene_recon.environment import environment
from scene_recon.run_files import read_json, write_atomically

logger = logging.getLogger(__name__)

COMPATIBILITY_VERSION = 3
"""Which version of the code a checkpoint's trained weights belong to.

Raise it by one whenever a code change means that a checkpoint written
before it no longer is what its config now describes:

* its weights would now be used differently, so they would give wrong
  results: for example, an encoder reading its features at other pixels, a
  renamed layer, or a network getting a different input;
* or the same config would now train differently: for example, some layers
  getting another learning rate, or a network another random start.

A checkpoint with another number is refused, and results made from it count
as out of date (see ``scene_recon.provenance``). Raising it retires every
existing checkpoint, so do it only when trained weights really are affected,
and add a line below.

1. Checkpoints written before this number existed.
2. Used only by intermediate commits of one review (2026-11-24), which each
   changed something listed under 3; no shared run was trained with it.
3. 2026-11-24: Hiera and tiny_conv features are read at their true pixel
   centres (commit 5925102); ConvNeXt V2's 1x1 fusion layer is renamed
   ``project`` -> ``adapter``, and every encoder's adapter trains at the base
   learning rate; pixel_aligned_nerf's decoder starts from random numbers of
   its own; the Hiera bridge's depth feature measures distance along the ray.
"""


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
    training_data: dict[str, Any],
    starting_weights: dict[str, Any],
) -> None:
    """Write everything needed to resume training or to evaluate the model.

    The checkpoint also stores the complete config it was trained with, the
    fingerprint of its training data (``SceneDataset.fingerprint``) and the
    ``COMPATIBILITY_VERSION``, which ``scene_recon.provenance`` checks before
    the checkpoint is used again; the pretrained weights the run started from
    (``pretrained_weights_of`` the model before its first step), which the
    suite report shows; and the software environment.

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
        "training_data": training_data,
        "starting_weights": starting_weights,
        "compatibility_version": COMPATIBILITY_VERSION,
        "random_state": _global_random_state(),
        "environment": environment(),
    }
    # The old progress file goes first: if saving is interrupted after the
    # checkpoint is replaced, read_progress reads the new checkpoint itself
    # instead of the progress of the old one.
    progress_file(path).unlink(missing_ok=True)
    write_atomically(path, lambda temporary_path: torch.save(state, temporary_path))
    write_atomically(
        progress_file(path), lambda temporary_path: temporary_path.write_text(json.dumps(asdict(progress)), encoding="utf-8")
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
    """The training progress stored with a checkpoint, or ``None`` if there is no checkpoint.

    Read from the small JSON file next to the checkpoint. If that file is
    missing or damaged (cut short, not JSON, a key given twice, or not the
    numbers ``save_checkpoint`` writes), the progress is read from the
    checkpoint itself instead, which takes longer.
    """
    if not checkpoint_path.is_file():
        return None

    sidecar = progress_file(checkpoint_path)
    if sidecar.is_file():
        progress = _recorded_progress(read_json(sidecar))
        if progress is not None:
            return progress
        logger.warning("%s is damaged; the training progress is read from %s instead.", sidecar, checkpoint_path.name)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    return TrainingProgress(step=state["step"], training_seconds=state["training_seconds"])


def _recorded_progress(content: dict[str, Any] | None) -> TrainingProgress | None:
    """The progress a JSON file records, or ``None`` if it is not exactly what ``save_checkpoint`` writes."""
    if content is None or set(content) != {field.name for field in fields(TrainingProgress)}:
        return None
    step, seconds = content["step"], content["training_seconds"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds >= 0:
        return None
    return TrainingProgress(step=step, training_seconds=float(seconds))


def read_checkpoint_config(path: Path) -> dict[str, Any]:
    """The config a checkpoint was trained with, read without loading the weights into memory."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)["config"]


def read_checkpoint_training_data(path: Path) -> dict[str, Any] | None:
    """The fingerprint of the data a checkpoint was trained on, or ``None`` for older checkpoints."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True).get("training_data")


def read_checkpoint_starting_weights(path: Path) -> dict[str, Any] | None:
    """The pretrained weights a checkpoint's run started from (``None`` if not recorded, as in older checkpoints)."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True).get("starting_weights")


def read_checkpoint_compatibility_version(path: Path) -> int:
    """The ``COMPATIBILITY_VERSION`` a checkpoint was written with (1 for checkpoints from before it existed)."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True).get("compatibility_version", 1)


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
