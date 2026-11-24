"""Which experiment produced a result, and whether a saved result is still valid.

A number in the suite report must describe the experiment that actually
produced it. This module answers three questions, for every command and for
the suite alike:

1. **May this checkpoint be used with this config?**
   ``check_checkpoint_matches_config``. Every checkpoint stores the config
   it was trained with. The settings that determine the trained weights (its
   *training identity*: seed, training data, model, loss, rendering,
   optimisation and batch settings) must be exactly the same in the config
   that resumes, evaluates or profiles it. Only the settings in
   ``CONTINUATION_SETTINGS`` may change, such as a larger step budget or
   another device. To train with different settings, start a new run.
   The checkpoint must also have been written by code with the current
   ``COMPATIBILITY_VERSION`` (see ``scene_recon.checkpoints``): the same
   config can mean different weights after a code change.

2. **Does a saved evaluation still describe the run?** ``evaluation_status``.
   Only if it was made from the very same checkpoint file, identified by the
   SHA-256 hash of its content, with the current measurement settings and
   the current ``COMPATIBILITY_VERSION``.

   A model without trainable parameters (frozen, or analytic) cannot be
   trained and has no checkpoint; it is evaluated as built, and its
   evaluation records the model's config and a hash of its weights instead
   (``untrained_model_record``). Both are needed: an analytic model may have
   no weights at all, so only its config tells two versions apart. The model
   is built seeded with ``experiment.seed`` (``build_seeded_model``), so that
   rebuilding it for this check gives the same weights, even if it fills
   frozen layers or buffers with random values.

3. **Does a saved profile still describe the run?** ``profile_status``.
   Only if it was made with the current model, rendering and profiling
   settings and ``COMPATIBILITY_VERSION``, on the same kind of device with
   the same software.

The suite uses 2 and 3 to decide what to redo, and the report uses them to
leave out-of-date results out of its tables and plots.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scene_recon.build import build_dataset, build_seeded_model, resolve_device
from scene_recon.checkpoints import (
    COMPATIBILITY_VERSION,
    TrainingProgress,
    read_checkpoint_compatibility_version,
    read_checkpoint_config,
    read_checkpoint_training_data,
)
from scene_recon.config import Config, flatten_settings, setting_in_sections
from scene_recon.data.base import SceneDataset
from scene_recon.environment import device_description
from scene_recon.registry import ConfigError
from scene_recon.run_files import RunFiles, read_json

CONTINUATION_SETTINGS = (
    "experiment.run_name",
    "experiment.output_dir",
    "experiment.device",
    "training.num_steps",
    "training.num_workers",
    "training.log_every",
    "training.validate_every",
    "training.checkpoint_every",
    "renderer.ray_chunk_size",  # only used to render whole images, never in training
    "validation",
    "evaluation",
    "profiling",
    "data.val",
    "data.test",
)
"""Settings that may differ between a checkpoint's training and its later use.

None of them changes the trained weights: they control how long training
runs, where and how fast it runs, and how the result is measured.
"""

_NOT_SET = "(not set)"


# ---------------------------------------------------------------- checkpoints


def training_identity(raw_config: dict[str, Any]) -> dict[str, Any]:
    """The settings of a config that determine the trained weights, as dotted names."""
    return {
        name: value
        for name, value in flatten_settings(raw_config).items()
        if not setting_in_sections(name, CONTINUATION_SETTINGS)
    }


def check_checkpoint_matches_config(checkpoint_path: Path, config: Config) -> None:
    """Refuse to continue, evaluate or profile a checkpoint with settings it was not trained with.

    Raises:
        ConfigError: If the checkpoint was written with another ``COMPATIBILITY_VERSION``,
            or listing every setting that differs, with the trained and the requested value.
    """
    check_checkpoint_is_compatible(checkpoint_path)
    trained = training_identity(read_checkpoint_config(checkpoint_path))
    requested = training_identity(config.raw)

    names = list(requested) + [name for name in trained if name not in requested]
    differences = [
        f"  {name}: trained with {_show(trained.get(name, _NOT_SET))}, requested {_show(requested.get(name, _NOT_SET))}"
        for name in names
        if _canonical(trained.get(name, _NOT_SET)) != _canonical(requested.get(name, _NOT_SET))
    ]
    if differences:
        raise ConfigError(
            f"{checkpoint_path} was trained with different settings than requested:\n"
            + "\n".join(differences)
            + "\nA checkpoint is only continued, evaluated or profiled with the settings it was trained with. "
            "To train with these settings, start a new run (another experiment.run_name, or another run "
            "in the suite). Settings that may change for an existing run: " + ", ".join(CONTINUATION_SETTINGS) + "."
        )


def check_checkpoint_is_compatible(checkpoint_path: Path) -> None:
    """Refuse a checkpoint whose trained weights the current code would use differently.

    Raises:
        ConfigError: If the checkpoint's ``COMPATIBILITY_VERSION`` is not the code's.
    """
    version = read_checkpoint_compatibility_version(checkpoint_path)
    if version == COMPATIBILITY_VERSION:
        return
    if version < COMPATIBILITY_VERSION:
        raise ConfigError(
            f"{checkpoint_path} was written by an older version of the code (compatibility version {version}, "
            f"the code is at {COMPATIBILITY_VERSION}). Since then the code uses trained weights differently, so "
            "this checkpoint would give wrong results; COMPATIBILITY_VERSION in scene_recon/checkpoints.py "
            f"says what changed. Train it again: delete {checkpoint_path.parent} or change experiment.run_name."
        )
    raise ConfigError(
        f"{checkpoint_path} was written by a newer version of the code (compatibility version {version}, "
        f"the code is at {COMPATIBILITY_VERSION}). Update the code to use it."
    )


def check_training_data_unchanged(checkpoint_path: Path, train_scenes: SceneDataset) -> None:
    """Refuse to continue training on data other than the data the checkpoint was trained on.

    The config can name the same directory while its content changed (e.g.
    a re-extracted or resized copy of the dataset); the fingerprint notices.
    """
    trained_on = read_checkpoint_training_data(checkpoint_path)
    if trained_on is None:  # a checkpoint from before fingerprints were recorded
        return
    current = _canonical(train_scenes.fingerprint())
    if set(trained_on) != set(current):
        # E.g. an SRN checkpoint from before content hashing ("files_sha256"):
        # its fingerprint cannot tell whether the content is the same, so the
        # current files cannot be trusted to be the ones it was trained on.
        raise ConfigError(
            f"{checkpoint_path} records its training data in an older fingerprint format "
            f"({_show(trained_on)}), which cannot confirm that the training split still holds the same "
            "content. It cannot be resumed or reused for training; start a new run to train. It can still "
            "be evaluated and profiled."
        )
    if _canonical(trained_on) != current:
        raise ConfigError(
            f"{checkpoint_path} was trained on different data than the training split now contains "
            f"(trained on {_show(trained_on)}, now {_show(current)}). Start a new run to train on the new data."
        )


def checkpoint_digest(path: Path) -> str:
    """SHA-256 of a checkpoint file's content, which identifies exactly which weights it holds."""
    status = path.stat()
    return _file_digest(str(path.resolve()), status.st_size, status.st_mtime_ns)


@functools.lru_cache(maxsize=64)
def _file_digest(path: str, size: int, modified_ns: int) -> str:
    # Size and modification time are part of the cache key only so that a
    # file is not hashed twice within one process; the digest is of the content.
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_weights_digest(model: Any) -> str:
    """SHA-256 of a model's saved state (parameters and persistent buffers): which weights it holds."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        values = tensor.detach().cpu().contiguous()
        digest.update(f"{name}\t{values.dtype}\t{tuple(values.shape)}\n".encode())
        digest.update(values.numpy().tobytes() if values.numel() else b"")
    return digest.hexdigest()


def untrained_model_record(config: Config, model: Any) -> dict[str, Any]:
    """What identifies a model evaluated as built: its config and its weights."""
    return _canonical({"config": config.raw["model"], "weights_sha256": model_weights_digest(model)})


def checkpoint_record(checkpoint_path: Path, progress: TrainingProgress) -> dict[str, Any]:
    """What a result file records about the checkpoint it was made from."""
    return {
        "path": str(checkpoint_path),
        "sha256": checkpoint_digest(checkpoint_path),
        "step": progress.step,
        "training_hours": progress.training_seconds / 3600,
    }


# ---------------------------------------------------------------- measurement settings


def evaluation_settings(config: Config) -> dict[str, Any]:
    """The settings and test data, besides the weights, that determine evaluation results."""
    return _canonical(
        {
            "compatibility_version": COMPATIBILITY_VERSION,
            "evaluation": config.raw["evaluation"],
            "renderer": config.raw["renderer"],
            "test_data": config.raw["data"]["test"],
            "test_data_fingerprint": build_dataset(config, "test").fingerprint(),
        }
    )


def profiling_settings(config: Config) -> dict[str, Any]:
    """The settings, workload and hardware that determine a speed measurement.

    The workload is identified by the test data itself, not only its config:
    the same config can point to images of another resolution, which changes
    how many rays a frame has.
    """
    test_scenes = build_dataset(config, "test")
    return _canonical(
        {
            "compatibility_version": COMPATIBILITY_VERSION,
            "profiling": config.raw["profiling"],
            "renderer": config.raw["renderer"],
            "model": config.raw["model"],
            "source_views": config.evaluation.source_view_sets[0],
            "test_data": config.raw["data"]["test"],
            "test_data_fingerprint": test_scenes.fingerprint(),
            "image_size": list(test_scenes.load_scene(0, [0]).image_size),
            "device": device_description(resolve_device(config.experiment.device)),
        }
    )


# ---------------------------------------------------------------- saved results


@dataclass(frozen=True)
class OutputStatus:
    """Whether a saved results file still describes its run."""

    state: str
    """``"current"``, ``"stale"`` (made from another checkpoint or with other
    settings), ``"unreadable"`` (damaged) or ``"missing"``."""

    reason: str
    """Why it is not current, in words. Empty when current."""

    content: dict[str, Any] | None
    """The file's content, if it could be read."""

    @property
    def is_current(self) -> bool:
        return self.state == "current"


def evaluation_status(config: Config) -> OutputStatus:
    """Whether the run's ``evaluation.json`` was made from its current checkpoint with the current settings."""
    files = RunFiles(config.run_dir)
    content, problem = _read_result(files.evaluation, required_keys=("checkpoint", "settings", "protocols"))
    if problem is not None:
        return problem
    if content["settings"] != evaluation_settings(config):
        return OutputStatus(
            "stale", "made with different evaluation, rendering or test-data settings, other test data, "
            "or an older version of the code", content
        )
    if content["checkpoint"] is None:
        # Made from a model without trainable parameters, as built (see run_evaluation).
        if files.checkpoint.is_file():
            return OutputStatus("stale", "made from the model as built, but the run now has a checkpoint", content)
        if content.get("untrained_model") != untrained_model_record(config, build_seeded_model(config)):
            return OutputStatus("stale", "made from a model built with other settings or weights", content)
        return OutputStatus("current", "", content)
    if not files.checkpoint.is_file():
        return OutputStatus("stale", "the run has no checkpoint", content)
    if content["checkpoint"].get("sha256") != checkpoint_digest(files.checkpoint):
        return OutputStatus("stale", "made from a different checkpoint than the run's current one", content)
    return OutputStatus("current", "", content)


def profile_status(config: Config) -> OutputStatus:
    """Whether the run's ``profile.json`` was made with the current settings on the current kind of device."""
    files = RunFiles(config.run_dir)
    content, problem = _read_result(files.profile, required_keys=("settings", "parameters", "rendering"))
    if problem is not None:
        return problem
    if content["settings"] != profiling_settings(config):
        return OutputStatus(
            "stale", "made with different settings, test data, device or software, or an older version of the code",
            content,
        )
    return OutputStatus("current", "", content)


def _read_result(path: Path, required_keys: tuple[str, ...]) -> tuple[dict[str, Any] | None, OutputStatus | None]:
    """Read a results file. Returns its content, or the status explaining why it cannot be used."""
    if not path.exists():
        return None, OutputStatus("missing", "not produced yet", None)
    content = read_json(path)
    if content is None or not all(key in content for key in required_keys):
        return None, OutputStatus("unreadable", f"{path.name} is damaged or from an older version", None)
    return content, None


def _canonical(value: Any) -> Any:
    """A JSON round trip, so values compare equal to what a results file records (lists, not tuples)."""
    return json.loads(json.dumps(value))


def _show(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)
