"""Loading and validating experiment configs.

A config is a YAML file. Three features keep configs short and safe:

* Inheritance. ``base: other.yaml`` loads ``other.yaml`` first (path relative
  to the current file) and merges the current file on top of it. ``base`` may
  also be a list of files, merged in order, which lets an experiment combine
  e.g. the common settings, a dataset file and a model file.
* Overrides. ``load_config(path, ["training.batch_size=8"])`` changes single
  values without editing a file; the command line exposes this as ``--set``.
* Validation. The pipeline's own settings are parsed into the dataclasses
  below. A misspelt or missing key is an error, never a silent default.

There are deliberately no default values in this file. Every setting is
spelled out in ``configs/base.yaml``, so the YAML files are the single place
to look when asking "what value did this run use?".

Sections that describe a swappable component (``model``, ``loss`` and each
entry of ``data``) are kept as plain dictionaries and handed to the matching
registry, because their keys depend on which component is selected.

Convention: a mapping with a ``name`` key always describes a component, and
``name`` selects its class. Do not use ``name`` as a key for anything else.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, get_type_hints

import yaml

from scene_recon.registry import ConfigError

DATA_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ExperimentSettings:
    run_name: str
    """Outputs are written to ``<output_dir>/<run_name>/``."""

    output_dir: str
    seed: int

    device: str
    """``"auto"`` (CUDA if available, otherwise CPU) or any torch device string."""


@dataclass(frozen=True)
class RendererSettings:
    num_coarse_samples: int
    """Evenly spread samples per ray."""

    num_fine_samples: int
    """Extra samples per ray placed where the coarse pass found content. 0 disables the fine pass."""

    background_color: list[float]
    """RGB in [0, 1] seen where a ray hits nothing. ShapeNet-SRN images have a white background."""

    ray_chunk_size: int
    """Rays rendered at once when rendering whole images. Lower it if evaluation runs out of memory."""


@dataclass(frozen=True)
class OptimizationSettings:
    optimizer: str
    """``"adam"`` or ``"adamw"``."""

    betas: list[float]
    weight_decay: float
    learning_rate: float

    warmup_steps: int
    """The learning rate rises linearly from zero over this many steps."""

    decay_rate: float
    """The learning rate is multiplied by this factor every ``decay_steps`` steps (smoothly)."""

    decay_steps: int

    group_overrides: dict[str, dict[str, Any]]
    """Per parameter-group ``learning_rate`` and/or ``warmup_steps``. See ``scene_recon.optim``."""


@dataclass(frozen=True)
class TrainingSettings:
    num_steps: int
    batch_size: int
    """Scenes per optimisation step."""

    rays_per_scene: int
    """Rays sampled from each scene per step. Rays per step = batch_size * rays_per_scene."""

    views_per_scene: int
    """Views loaded per scene per step. Source views are chosen among them and rays are drawn from all of them."""

    num_source_views: list[int]
    """Possible numbers of source views. One is picked at random for every step."""

    num_workers: int
    log_every: int
    validate_every: int
    checkpoint_every: int


@dataclass(frozen=True)
class ValidationSettings:
    """The quick check on the validation split that runs during training."""

    num_scenes: int
    num_target_views: int
    source_views: list[int]
    """View indices given to the model as input."""

    metrics: list[str]


@dataclass(frozen=True)
class EvaluationSettings:
    """The full protocol on the test split, run by ``scene-recon evaluate``."""

    source_view_sets: list[list[int]]
    """One evaluation pass is run per entry, e.g. ``[[64], [64, 104]]`` for 1 and 2 input views."""

    metrics: list[str]

    num_scenes: int | None
    """Limit the number of test scenes. ``null`` evaluates all of them."""

    num_target_views: int | None
    """Limit the number of rendered views per scene. ``null`` renders every non-source view."""

    save_images: bool
    """Save every rendered view under ``<run_dir>/renders/``. Takes a lot of disk space on the full test set."""

    num_preview_scenes: int
    """Scenes whose first rendered view is always saved, for side-by-side comparison of runs."""


@dataclass(frozen=True)
class ProfilingSettings:
    """Rendering-speed measurement, run by ``scene-recon profile``."""

    num_frames: int
    """Timed full-frame renders; the reported speed is their average."""

    num_warmup_frames: int
    """Untimed renders done first, so one-off start-up costs are not counted."""


@dataclass(frozen=True)
class Config:
    """A fully validated experiment config."""

    experiment: ExperimentSettings
    data: dict[str, dict[str, Any]]
    """One dataset component config per split: ``train``, ``val`` and ``test``."""

    model: dict[str, Any]
    loss: dict[str, Any]
    renderer: RendererSettings
    optimization: OptimizationSettings
    training: TrainingSettings
    validation: ValidationSettings
    evaluation: EvaluationSettings
    profiling: ProfilingSettings

    raw: dict[str, Any]
    """The merged YAML content, saved next to the run's outputs for reproducibility."""

    @property
    def run_dir(self) -> Path:
        return Path(self.experiment.output_dir) / self.experiment.run_name


_SETTINGS_SECTIONS = {
    "experiment": ExperimentSettings,
    "renderer": RendererSettings,
    "optimization": OptimizationSettings,
    "training": TrainingSettings,
    "validation": ValidationSettings,
    "evaluation": EvaluationSettings,
    "profiling": ProfilingSettings,
}
_COMPONENT_SECTIONS = ("data", "model", "loss")


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    """Read a YAML config, resolve inheritance, apply overrides and validate.

    Args:
        path: The YAML file to load.
        overrides: ``"dotted.key=value"`` strings, applied in order. Values
            are parsed as YAML, so ``"model.hidden_width=64"`` yields an int
            and ``"evaluation.metrics=[psnr, ssim]"`` yields a list.
    """
    raw = load_yaml_with_bases(Path(path))
    for override in overrides:
        apply_override(raw, override)
    return parse_config(raw)


def parse_config(raw: Mapping[str, Any]) -> Config:
    """Validate a merged config dictionary and convert it to a ``Config``."""
    expected_sections = set(_SETTINGS_SECTIONS) | set(_COMPONENT_SECTIONS)
    _check_keys(raw.keys(), expected_sections, where="the config")

    settings = {
        section: _parse_settings(settings_class, raw[section], section)
        for section, settings_class in _SETTINGS_SECTIONS.items()
    }

    _check_keys(raw["data"].keys(), set(DATA_SPLITS), where="'data'")
    _check_source_view_sets(settings["evaluation"].source_view_sets)

    return Config(
        data=copy.deepcopy(dict(raw["data"])),
        model=copy.deepcopy(dict(raw["model"])),
        loss=copy.deepcopy(dict(raw["loss"])),
        raw=copy.deepcopy(dict(raw)),
        **settings,
    )


def save_config(config: Config, path: str | Path) -> None:
    """Write the merged config back to disk as a single self-contained YAML file."""
    with open(path, "w") as file:
        yaml.safe_dump(config.raw, file, sort_keys=False)


def load_yaml_with_bases(path: Path) -> dict[str, Any]:
    """Load a YAML file and merge it on top of the file(s) named by its ``base`` key, if any."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as file:
        content = yaml.safe_load(file) or {}
    if not isinstance(content, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level.")
    return resolve_bases(content, relative_to=path.parent)


def resolve_bases(content: Mapping[str, Any], relative_to: Path) -> dict[str, Any]:
    """Merge a config fragment on top of the file(s) named by its ``base`` key.

    Args:
        content: A config fragment, possibly with a ``base`` key holding one
            path or a list of paths.
        relative_to: Directory that relative ``base`` paths start from.

    Returns:
        The merged content, without the ``base`` key.
    """
    content = dict(content)
    base_paths = content.pop("base", [])
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    merged: dict[str, Any] = {}
    for base_path in base_paths:
        merged = merge_configs(merged, load_yaml_with_bases(relative_to / base_path))
    return merge_configs(merged, content)


def merge_configs(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into ``base``.

    Nested mappings are merged key by key. A component section that selects a
    different ``name`` replaces the base section entirely instead, because
    the constructor arguments of one component mean nothing to another.
    """
    merged = dict(base)
    for key, value in update.items():
        both_are_mappings = isinstance(value, Mapping) and isinstance(merged.get(key), Mapping)
        if both_are_mappings and not _selects_different_component(merged[key], value):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _selects_different_component(base: Mapping[str, Any], update: Mapping[str, Any]) -> bool:
    return "name" in base and "name" in update and base["name"] != update["name"]


def apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply one ``"dotted.key=value"`` override in place."""
    dotted_key, separator, value_text = override.partition("=")
    if not separator:
        raise ConfigError(f"Override '{override}' is not of the form key=value.")

    *parent_keys, last_key = dotted_key.strip().split(".")
    section = raw
    for key in parent_keys:
        section = section.setdefault(key, {})
        if not isinstance(section, dict):
            raise ConfigError(f"Cannot apply override '{override}': '{key}' is not a mapping.")
    section[last_key] = yaml.safe_load(value_text)


def _check_source_view_sets(source_view_sets: list[list[int]]) -> None:
    """Each evaluation protocol must use a different number of views.

    Results are labelled and plotted by the number of input views ("1 view",
    "2 views"), so two protocols with the same number would be indistinguishable.
    """
    sizes = [len(views) for views in source_view_sets]
    if len(set(sizes)) != len(sizes):
        raise ConfigError(
            f"evaluation.source_view_sets {source_view_sets} contains two sets with the same number of "
            "views. Results are identified by the number of input views, so each set must have a different size."
        )
    if 0 in sizes:
        raise ConfigError("evaluation.source_view_sets must not contain an empty set.")


def _check_keys(actual: Any, expected: set[str], where: str) -> None:
    """Raise a readable error if a mapping has unknown or missing keys."""
    actual = set(actual)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where}: {unknown}. Valid keys: {sorted(expected)}.")
    if missing:
        raise ConfigError(f"Missing key(s) in {where}: {missing}. See configs/base.yaml for an example.")


def _parse_settings(settings_class: type, values: Mapping[str, Any], section: str) -> Any:
    """Build one settings dataclass from its config section."""
    if not isinstance(values, Mapping):
        raise ConfigError(f"Config section '{section}' must be a mapping.")

    type_hints = get_type_hints(settings_class)
    field_names = {field.name for field in dataclasses.fields(settings_class)}
    _check_keys(values.keys(), field_names, where=f"'{section}'")

    parsed = {}
    for name in field_names:
        parsed[name] = _coerce_number(values[name], type_hints[name], where=f"{section}.{name}")
    return settings_class(**parsed)


def _coerce_number(value: Any, expected_type: Any, where: str) -> Any:
    """Convert a value to int or float when the setting is declared as one.

    This exists mainly because YAML reads ``1e-4`` (no decimal point) as a
    string; without the conversion that typo would surface much later as a
    confusing error inside the optimiser.
    """
    if expected_type not in (int, float) or isinstance(value, bool):
        return value
    try:
        return expected_type(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"'{where}' must be a number, got {value!r}.") from error
