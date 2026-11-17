"""Turning a validated ``Config`` into the objects that make up an experiment."""

from __future__ import annotations

import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

# Importing these packages runs the @register decorators of everything in them.
import scene_recon.data  # noqa: F401
import scene_recon.losses  # noqa: F401
import scene_recon.metrics  # noqa: F401
import scene_recon.models  # noqa: F401
from scene_recon.config import Config
from scene_recon.data.base import SceneDataset
from scene_recon.models.base import ReconstructionModel
from scene_recon.registry import DATASETS, LOSSES, METRICS, MODELS, ConfigError


def resolve_device(name: str) -> torch.device:
    """Translate the ``experiment.device`` setting into a torch device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Seed every random number generator the pipeline draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config: Config) -> ReconstructionModel:
    model = MODELS.build(config.model)
    if not isinstance(model, ReconstructionModel):
        raise ConfigError(
            f"Model '{config.model['name']}' ({type(model).__name__}) must subclass ReconstructionModel."
        )
    return model


def build_dataset(config: Config, split: str) -> SceneDataset:
    """Build the dataset of one split: ``"train"``, ``"val"`` or ``"test"``."""
    return DATASETS.build(config.data[split])


def build_loss(config: Config) -> Callable[..., dict[str, torch.Tensor]]:
    return LOSSES.build(config.loss)


def build_metrics(entries: Sequence[str | Mapping[str, Any]]) -> dict[str, nn.Module]:
    """Build the metrics named in a config list.

    Each entry is either a metric name (``psnr``) or, to pass constructor
    arguments, a mapping with a ``name`` key (``{name: lpips, network: alex}``).

    Returns:
        The metrics keyed by name, in the order they were listed.
    """
    metrics = {}
    for entry in entries:
        metric_config = {"name": entry} if isinstance(entry, str) else entry
        metrics[metric_config["name"]] = METRICS.build(metric_config)
    return metrics
