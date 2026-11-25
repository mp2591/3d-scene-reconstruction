"""Keeping a comparison between runs fair.

Two kinds of settings are treated differently:

* Measurement settings define how a run is scored: the evaluation protocol,
  validation, profiling, the validation and test data, and two settings on
  which the measured rendering speed depends: the number of rays rendered at
  once (``renderer.ray_chunk_size``) and the precision of float32 matrix
  products (``experiment.float32_matmul_precision``). If they differed, the runs' numbers would not be
  comparable at all, so every run of a suite must use the same ones.
  ``check_identical_measurement`` enforces this.

* Every other setting may differ between runs; that is what an experiment
  varies (the model, an encoder's learning rate, a renderer ablation...).
  ``differing_settings`` lists them, and the suite report shows the list, so
  a reader sees exactly what was varied and that everything else was held fixed.

The training data itself is identical for runs that share the settings in
``TRAINING_DATA_SETTINGS`` (see ``scene_recon.data.schedule``).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from scene_recon.config import Config, flatten_settings, setting_in_sections
from scene_recon.registry import ConfigError

MEASUREMENT_SETTINGS = (
    "evaluation",
    "validation",
    "profiling",
    "data.val",
    "data.test",
    "renderer.ray_chunk_size",
    "experiment.float32_matmul_precision",  # rendering speed depends on it, as on ray_chunk_size
)
"""Config sections that must be identical for every run of a suite."""

TRAINING_DATA_SETTINGS = (
    "experiment.seed",
    "data.train",
    "training.batch_size",
    "training.views_per_scene",
    "training.num_source_views",
    "training.rays_per_scene",
)
"""Settings that together determine the training data of every step."""

# Settings that differ between runs by construction, and are not worth listing.
_RUN_IDENTITY = ("experiment.run_name", "experiment.output_dir")

MISSING = "–"
"""Shown for a setting that one run has and another does not (e.g. after swapping a component)."""


def check_identical_measurement(configs: Mapping[str, Config]) -> None:
    """Raise if any measurement setting differs between the runs.

    Raises:
        ConfigError: Naming every differing setting and the value each run uses.
    """
    differences = {
        setting: values
        for setting, values in differing_settings(configs).items()
        if setting_in_sections(setting, MEASUREMENT_SETTINGS)
    }
    if not differences:
        return

    lines = [
        "Settings that decide how runs are measured must be the same for every run of a suite;",
        "set them in 'common' rather than in individual runs. These differ:",
    ]
    for setting, values in differences.items():
        per_run = ", ".join(f"{run}={format_value(value)}" for run, value in values.items())
        lines.append(f"  {setting}: {per_run}")
    raise ConfigError("\n".join(lines))


def differing_settings(configs: Mapping[str, Config]) -> dict[str, dict[str, Any]]:
    """Every setting whose value is not the same for all runs.

    Returns:
        Dotted setting name -> run name -> value, in config order. A run
        without the setting has the value ``MISSING``.
    """
    flattened = {run: flatten_settings(config.raw) for run, config in configs.items()}
    all_settings: list[str] = []
    for settings in flattened.values():
        all_settings.extend(name for name in settings if name not in all_settings)

    differences = {}
    for setting in all_settings:
        if setting_in_sections(setting, _RUN_IDENTITY):
            continue
        values = {run: settings.get(setting, MISSING) for run, settings in flattened.items()}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) > 1:
            differences[setting] = values
    return differences


def trained_on_identical_data(differences: Mapping[str, Any]) -> bool:
    """Whether none of the differing settings changes the training data."""
    return not any(setting_in_sections(setting, TRAINING_DATA_SETTINGS) for setting in differences)


def format_value(value: Any) -> str:
    """A compact, readable form of a setting's value."""
    return value if isinstance(value, str) else json.dumps(value)
