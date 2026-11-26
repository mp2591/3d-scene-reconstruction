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
  Every value is checked against its declared type (a whole number must be a
  whole number, ``true`` is not a number) and against the rules in
  ``_VALUE_RULES`` and ``_check_values``, such as "at least one training step".

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
import math
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Union, get_args, get_origin, get_type_hints

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

    float32_matmul_precision: str
    """How float32 matrix products are computed, passed to ``torch.set_float32_matmul_precision``:
    ``"highest"`` (full float32), ``"high"`` (TensorFloat-32 where the GPU has it) or ``"medium"``."""


FLOAT32_MATMUL_PRECISIONS = ("highest", "high", "medium")


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

    first_phase_steps: int
    """Steps at the start of training that use the two settings below instead (pixelNeRF: 300,000; 0 = none)."""

    first_phase_num_source_views: list[int]
    """Possible numbers of source views during the first phase. One is picked at random for every step."""

    first_phase_rays_in_object_box: bool
    """During the first phase, draw rays only from the smallest box around each view's object (its
    pixels that differ from the background colour in every channel), as pixelNeRF does: most of an
    image is empty."""

    num_workers: int
    log_every: int
    validate_every: int
    checkpoint_every: int

    def source_view_counts(self) -> list[int]:
        """Every number of source views that training uses, in either phase."""
        counts = set(self.num_source_views)
        if self.first_phase_steps > 0:
            counts |= set(self.first_phase_num_source_views)
        return sorted(counts)


@dataclass(frozen=True)
class ValidationSettings:
    """The quick check on the validation split that runs during training."""

    num_scenes: int
    num_target_views: int
    source_views: list[int]
    """View indices given to the model as input."""

    metrics: list[str | dict[str, Any]]
    """Metric names (``psnr``), or mappings with constructor options (``{name: lpips, network: alex}``)."""


@dataclass(frozen=True)
class EvaluationSettings:
    """The full protocol on the test split, run by ``scene-recon evaluate``."""

    source_view_sets: list[list[int]]
    """One evaluation pass is run per entry, e.g. ``[[64], [64, 104]]`` for 1 and 2 input views."""

    metrics: list[str | dict[str, Any]]
    """Metric names (``psnr``), or mappings with constructor options (``{name: lpips, network: alex}``)."""

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

    min_free_gpu_memory: float
    """Share of the GPU's memory that must stay free while rendering, or the speed is refused.

    Some drivers (on Windows) move GPU memory to the computer's main memory
    when the GPU is full instead of failing, and rendering gets several times
    slower. Memory used by other programs counts too. The free memory is read
    after each frame is queued and once all frames have run; memory another
    program takes and releases between two readings is missed.
    """


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
    settings["optimization"] = dataclasses.replace(
        settings["optimization"], group_overrides=_parse_group_overrides(settings["optimization"].group_overrides)
    )
    _check_values(settings)

    return Config(
        data=copy.deepcopy(dict(raw["data"])),
        model=copy.deepcopy(dict(raw["model"])),
        loss=copy.deepcopy(dict(raw["loss"])),
        raw=copy.deepcopy(dict(raw)),
        **settings,
    )


def save_config(config: Config, path: str | Path) -> None:
    """Write the merged config back to disk as a single self-contained YAML file."""
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config.raw, file, sort_keys=False)


def load_yaml_with_bases(path: Path) -> dict[str, Any]:
    """Load a YAML file and merge it on top of the file(s) named by its ``base`` key, if any."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as file:
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
        parsed[name] = _convert(values[name], type_hints[name], where=f"{section}.{name}")
    return settings_class(**parsed)


def _convert(value: Any, expected_type: Any, where: str) -> Any:
    """Check a value against a setting's declared type, converting where that is unambiguous.

    Conversions: a whole-number float such as ``4.0e+5`` becomes an int, and
    a numeric string becomes a float (YAML reads ``1e-4``, without a decimal
    point, as a string). Everything else must already have the right type.
    """
    origin = get_origin(expected_type)

    if expected_type is Any:
        return value
    if origin in (Union, types.UnionType):  # e.g. int | None, or str | dict[str, Any]
        options = get_args(expected_type)
        if value is None and type(None) in options:
            return None
        alternatives = [option for option in options if option is not type(None)]
        for option in alternatives:
            try:
                return _convert(value, option, where)
            except ConfigError:
                continue
        expected = " or ".join(_describe_type(option) for option in alternatives)
        raise ConfigError(f"'{where}' must be {expected}, got {value!r}.")
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"'{where}' must be a list, got {value!r}.")
        (item_type,) = get_args(expected_type)
        return [_convert(item, item_type, f"{where}[{index}]") for index, item in enumerate(value)]
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"'{where}' must be a mapping, got {value!r}.")
        return dict(value)
    if expected_type is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{where}' must be true or false, got {value!r}.")
        return value
    if expected_type is int:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{where}' must be a whole number, got {value!r}.")
        return value
    if expected_type is float:
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(f"'{where}' must be a finite number, got {value!r}.")
        return float(value)
    if expected_type is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{where}' must be text, got {value!r}.")
        return value
    raise TypeError(f"Settings of type {expected_type} are not supported by the config parser.")


def _describe_type(expected_type: Any) -> str:
    """How a setting type is named in error messages."""
    origin = get_origin(expected_type) or expected_type
    names = {
        str: "text",
        int: "a whole number",
        float: "a number",
        bool: "true or false",
        list: "a list",
        dict: "a mapping",
    }
    return names.get(origin, str(expected_type))


GROUP_OVERRIDE_SETTINGS = {"learning_rate": float, "warmup_steps": int}
"""The settings a parameter group can override in ``optimization.group_overrides``, with their types."""


def _parse_group_overrides(overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Check every group override with the same rules as the setting it overrides.

    A learning rate must be a finite positive number and a warmup a whole
    number of steps, zero or more; anything else is reported, all at once.
    """
    parsed, problems = {}, []
    for group, values in overrides.items():
        where = f"optimization.group_overrides.{group}"
        if not isinstance(group, str) or not isinstance(values, Mapping):
            problems.append(f"{where} must map a parameter group name to its settings, got {values!r}.")
            continue
        unknown = sorted(set(values) - set(GROUP_OVERRIDE_SETTINGS))
        if unknown:
            problems.append(
                f"{where} has unknown key(s) {unknown}; a group can override {sorted(GROUP_OVERRIDE_SETTINGS)}."
            )
            continue
        parsed[group] = {}
        for name, value in values.items():
            try:
                converted = _convert(value, GROUP_OVERRIDE_SETTINGS[name], where=f"{where}.{name}")
            except ConfigError as error:
                problems.append(str(error))
                continue
            if name == "learning_rate" and converted <= 0:
                problems.append(f"{where}.learning_rate must be positive, got {converted}.")
            if name == "warmup_steps" and converted < 0:
                problems.append(f"{where}.warmup_steps must be at least 0, got {converted}.")
            parsed[group][name] = converted
    if problems:
        raise ConfigError("Invalid setting(s):\n  " + "\n  ".join(problems))
    return parsed


# Smallest allowed value of numeric settings. A setting that is None (where
# allowed, e.g. evaluation.num_scenes) is not checked.
_VALUE_RULES: dict[str, int] = {
    "renderer.num_coarse_samples": 2,
    "renderer.num_fine_samples": 0,
    "renderer.ray_chunk_size": 1,
    "optimization.weight_decay": 0,
    "optimization.warmup_steps": 0,
    "optimization.decay_steps": 1,
    "training.num_steps": 1,
    "training.batch_size": 1,
    "training.rays_per_scene": 1,
    "training.views_per_scene": 1,
    "training.first_phase_steps": 0,
    "training.num_workers": 0,
    "training.log_every": 1,
    "training.validate_every": 1,
    "training.checkpoint_every": 1,
    "validation.num_scenes": 1,
    "validation.num_target_views": 1,
    "evaluation.num_scenes": 1,
    "evaluation.num_target_views": 1,
    "evaluation.num_preview_scenes": 0,
    "profiling.num_frames": 1,
    "profiling.num_warmup_frames": 0,
    "profiling.min_free_gpu_memory": 0,
}


def _check_values(settings: dict[str, Any]) -> None:
    """Check that the settings' values make sense, and report every problem at once."""
    problems = []

    precision = settings["experiment"].float32_matmul_precision
    if precision not in FLOAT32_MATMUL_PRECISIONS:
        problems.append(
            f"experiment.float32_matmul_precision must be one of {', '.join(FLOAT32_MATMUL_PRECISIONS)}, "
            f"got {precision!r}."
        )

    for setting, minimum in _VALUE_RULES.items():
        section, name = setting.split(".")
        value = getattr(settings[section], name)
        if value is not None and value < minimum:
            problems.append(f"{setting} must be at least {minimum}, got {value}.")

    if settings["profiling"].min_free_gpu_memory >= 1:
        problems.append(
            "profiling.min_free_gpu_memory is a share of the GPU's memory and must be below 1, "
            f"got {settings['profiling'].min_free_gpu_memory}."
        )

    renderer = settings["renderer"]
    if renderer.num_fine_samples > 0 and renderer.num_coarse_samples < 3:
        problems.append(
            "renderer.num_coarse_samples must be at least 3 when the fine pass is enabled "
            "(fine samples are drawn between coarse samples)."
        )
    if len(renderer.background_color) != 3 or not all(0 <= value <= 1 for value in renderer.background_color):
        problems.append("renderer.background_color must be three values between 0 and 1.")

    optimization = settings["optimization"]
    if optimization.learning_rate <= 0:
        problems.append("optimization.learning_rate must be positive.")
    if not 0 < optimization.decay_rate <= 1:
        problems.append("optimization.decay_rate must be greater than 0 and at most 1.")
    if len(optimization.betas) != 2 or not all(0 <= beta < 1 for beta in optimization.betas):
        problems.append("optimization.betas must be two values in [0, 1).")

    for name in ("num_source_views", "first_phase_num_source_views"):
        counts = getattr(settings["training"], name)
        if not counts or min(counts) < 1:
            problems.append(f"training.{name} must be a non-empty list of positive numbers.")

    problems += _view_list_problems(settings["validation"].source_views, "validation.source_views")
    problems += _metric_list_problems(settings["validation"].metrics, "validation.metrics")

    evaluation = settings["evaluation"]
    problems += _metric_list_problems(evaluation.metrics, "evaluation.metrics")
    if not evaluation.source_view_sets:
        problems.append("evaluation.source_view_sets must contain at least one set of views.")
    for index, views in enumerate(evaluation.source_view_sets):
        problems += _view_list_problems(views, f"evaluation.source_view_sets[{index}]")
    sizes = [len(views) for views in evaluation.source_view_sets]
    if len(set(sizes)) != len(sizes):
        # Results are labelled and plotted by the number of input views, so
        # two sets of the same size could not be told apart.
        problems.append(
            f"evaluation.source_view_sets {evaluation.source_view_sets} has two sets with the same number of "
            "views; each set must have a different size, because results are identified by it."
        )

    if problems:
        raise ConfigError("Invalid setting(s):\n  " + "\n  ".join(problems))


def _metric_list_problems(entries: list[str | dict[str, Any]], where: str) -> list[str]:
    """A metric list must be non-empty, name every metric, and not list a metric twice.

    Whether the names and options fit a registered metric is checked when the
    metrics are built (in a suite's preflight, before any training).
    """
    if not entries:
        return [f"{where} must name at least one metric."]
    problems, names = [], []
    for index, entry in enumerate(entries):
        name = entry if isinstance(entry, str) else entry.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"{where}[{index}] must have a 'name' naming the metric, got {entry!r}.")
        else:
            names.append(name)
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        problems.append(f"{where} lists {repeated} more than once; results are stored by metric name.")
    return problems


def _view_list_problems(views: list[int], where: str) -> list[str]:
    """A list of view indices must be non-empty, non-negative and free of repeats."""
    if not views:
        return [f"{where} must contain at least one view index."]
    problems = []
    if min(views) < 0:
        problems.append(f"{where} must not contain negative view indices, got {views}.")
    if len(set(views)) != len(views):
        problems.append(f"{where} lists a view more than once, got {views}.")
    return problems


def flatten_settings(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Turn nested mappings into dotted names, e.g. ``{"a": {"b": 1}}`` -> ``{"a.b": 1}``.

    Lists and other values are kept whole.
    """
    flat = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(flatten_settings(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def setting_in_sections(setting: str, sections: Sequence[str]) -> bool:
    """Whether a dotted setting name is one of ``sections`` or lies inside one of them.

    Example: ``"data.test.split_dir"`` is in ``("data.test",)``; ``"data.train"`` is not.
    """
    return any(setting == section or setting.startswith(section + ".") for section in sections)
