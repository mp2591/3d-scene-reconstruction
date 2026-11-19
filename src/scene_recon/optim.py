"""Building the optimiser and the learning-rate schedule from config.

Learning rates are organised around the model's parameter groups (see
``ReconstructionModel.parameter_groups``). For a group, at a given step::

    lr = configured learning rate      (optimization.learning_rate, or a group override)
         * the group's lr_scale        (set by the model, e.g. for layer-wise decay)
         * warmup factor               (rises linearly from 0 to 1 over warmup_steps)
         * decay factor                (decay_rate ** (step / decay_steps))

``optimization.group_overrides`` changes ``learning_rate`` and/or
``warmup_steps`` for individual groups. An override applies to the group of
that name and to every group nested under it, so::

    group_overrides:
      encoder: {learning_rate: 1.0e-5, warmup_steps: 2000}

affects ``encoder``, ``encoder.stage0``, ``encoder.stage1``, and so on. When
several overrides match a group, the most specific (longest) one wins.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from scene_recon.config import GROUP_OVERRIDE_SETTINGS, OptimizationSettings
from scene_recon.models.base import ReconstructionModel
from scene_recon.models.contract import check_parameter_groups
from scene_recon.registry import ConfigError

_OPTIMIZER_CLASSES = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def build_optimizer_and_scheduler(
    model: ReconstructionModel, settings: OptimizationSettings
) -> tuple[Optimizer, LambdaLR]:
    """Create the optimiser with one torch parameter group per model parameter group.

    The scheduler must be stepped once after every optimiser step.

    Raises:
        ModelContractError: If a trainable parameter is in no group, or in several.
        ConfigError: If the optimiser settings do not fit the model's groups.
    """
    if settings.optimizer not in _OPTIMIZER_CLASSES:
        raise ConfigError(
            f"Unknown optimizer '{settings.optimizer}'. Available: {sorted(_OPTIMIZER_CLASSES)}."
        )

    check_parameter_groups(model)
    groups = model.parameter_groups()
    _check_overrides(settings.group_overrides, [group.name for group in groups])

    torch_groups = []
    schedules = []
    for group in groups:
        group_settings = _settings_for_group(group.name, settings)
        torch_groups.append(
            {
                "name": group.name,
                "params": group.parameters,
                "lr": group_settings["learning_rate"] * group.lr_scale,
            }
        )
        schedules.append(
            _make_schedule(group_settings["warmup_steps"], settings.decay_rate, settings.decay_steps)
        )

    optimizer = _OPTIMIZER_CLASSES[settings.optimizer](
        torch_groups, betas=tuple(settings.betas), weight_decay=settings.weight_decay
    )
    scheduler = LambdaLR(optimizer, lr_lambda=schedules)
    return optimizer, scheduler


def learning_rate_factor(step: int, warmup_steps: int, decay_rate: float, decay_steps: int) -> float:
    """The factor by which the base learning rate is multiplied at a given step.

    Combines a linear warmup with a smooth exponential decay. With
    ``warmup_steps=0`` and ``decay_rate=1.0`` the factor is always 1.
    """
    warmup = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
    decay = decay_rate ** (step / decay_steps)
    return warmup * decay


def _make_schedule(warmup_steps: int, decay_rate: float, decay_steps: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return learning_rate_factor(step, warmup_steps, decay_rate, decay_steps)

    return schedule


def _settings_for_group(group_name: str, settings: OptimizationSettings) -> dict[str, Any]:
    """Resolve the learning rate and warmup of one group, applying the best-matching override."""
    resolved = {"learning_rate": settings.learning_rate, "warmup_steps": settings.warmup_steps}

    matching_overrides = [key for key in settings.group_overrides if _override_applies(key, group_name)]
    if matching_overrides:
        # The values were checked and converted when the config was parsed.
        most_specific = max(matching_overrides, key=len)
        resolved.update(settings.group_overrides[most_specific])
    return resolved


def _override_applies(override_key: str, group_name: str) -> bool:
    return group_name == override_key or group_name.startswith(override_key + ".")


def _check_overrides(overrides: dict[str, dict[str, Any]], group_names: list[str]) -> None:
    """Reject overrides that would silently do nothing or that name unknown settings.

    The override values themselves are checked when the config is parsed.
    """
    for key, values in overrides.items():
        if not any(_override_applies(key, name) for name in group_names):
            raise ConfigError(
                f"optimization.group_overrides has an entry for '{key}', but the model has no such "
                f"parameter group. Its groups are: {group_names}."
            )
        unknown = set(values) - set(GROUP_OVERRIDE_SETTINGS)
        if unknown:
            raise ConfigError(
                f"optimization.group_overrides.{key} has unknown key(s) {sorted(unknown)}. "
                f"A group can override: {sorted(GROUP_OVERRIDE_SETTINGS)}."
            )
