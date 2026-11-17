"""Running a set of experiments from one suite file.

A suite file lists runs that share most of their settings, for example the
architectures being compared in the proposal::

    suite_name: proposal_comparison
    stages: [train, evaluate, profile]

    common:                       # merged into every run
      base: [../base.yaml, ../data/srn_cars.yaml]

    runs:                         # executed in this order
      pixelnerf_resnet34:
        base: ../models/pixelnerf_resnet34.yaml
      hiera_maskunit:
        base: ../models/hiera_maskunit.yaml
        optimization:
          group_overrides: {encoder: {learning_rate: 1.0e-5}}

``common`` and each entry of ``runs`` are ordinary config fragments: they may
use ``base`` (paths relative to the suite file) and any config section. A
run's config is ``common``, then the run's own fragment, then any ``--set``
overrides from the command line. The run's key becomes its
``experiment.run_name``, and all runs write into
``<output_dir>/<suite_name>/``, with ``output_dir`` taken from
``common``'s ``experiment.output_dir``.

Running a suite:

1. Every run's config is built and validated, its datasets are opened and its
   model is checked against the pipeline's contract. The runs must also use
   identical measurement settings (see ``scene_recon.fairness``). Any problem
   stops the suite here, before hours are spent on the first run.
2. The runs are executed one after another, each through the requested
   stages. Work that is already done is skipped (see ``_run_stages``), so an
   interrupted suite is resumed by starting it again.
3. After every run the suite report (tables and plots) is rewritten, so
   partial results can be inspected while later runs are still training.

If a run fails, the error is logged and the suite moves on to the next run.
"""

from __future__ import annotations

import gc
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from scene_recon.build import build_dataset, build_model, resolve_device
from scene_recon.checkpoints import read_progress
from scene_recon.config import (
    DATA_SPLITS,
    Config,
    apply_override,
    merge_configs,
    parse_config,
    resolve_bases,
)
from scene_recon.fairness import check_identical_measurement
from scene_recon.models.contract import check_model_contract
from scene_recon.registry import ConfigError
from scene_recon.report import write_report
from scene_recon.run_files import RunFiles
from scene_recon.stages import (
    check_evaluation_settings,
    dataset_image_size,
    evaluation_settings_snapshot,
    profiling_settings_snapshot,
    run_evaluation,
    run_profiling,
    run_training,
)
from scene_recon.training import check_training_settings

logger = logging.getLogger(__name__)

STAGES = ("train", "evaluate", "profile")
"""All stages, in the order they are executed."""

_SUITE_KEYS = {"suite_name", "stages", "common", "runs"}
_VALID_NAME = re.compile(r"^[A-Za-z0-9_.+-]+$")


class PreflightError(Exception):
    """Raised when one or more runs of a suite cannot start. The message lists every problem found."""


@dataclass(frozen=True)
class Suite:
    name: str
    stages: list[str]
    """The stages to execute, in execution order."""

    runs: dict[str, Config]
    """Every run's validated config, in execution order."""

    directory: Path
    """Where the runs and the suite report are written: ``<output_dir>/<suite_name>/``."""


def load_suite(path: str | Path, overrides: Sequence[str] = ()) -> Suite:
    """Read a suite file and build the config of every run.

    Args:
        path: The suite YAML file.
        overrides: ``"dotted.key=value"`` strings applied to every run, e.g.
            ``"training.num_steps=100"`` for a quick trial of the whole suite.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Suite file not found: {path}")
    with open(path) as file:
        content = yaml.safe_load(file) or {}

    _check_suite_keys(content, path)
    suite_name = _checked_name(content["suite_name"], "suite_name")
    stages = _checked_stages(content["stages"])
    runs = content["runs"]
    if not isinstance(runs, Mapping) or not runs:
        raise ConfigError(f"'runs' in {path} must be a non-empty mapping from run name to config.")

    common = resolve_bases(content["common"] or {}, relative_to=path.parent)
    output_dir = common.get("experiment", {}).get("output_dir")
    if output_dir is None:
        raise ConfigError(f"'common' in {path} must set experiment.output_dir (usually through base.yaml).")
    suite_directory = Path(output_dir) / suite_name

    # Every run is checked before reporting, so that one attempt lists all mistakes.
    configs, problems = {}, []
    for run_name, fragment in runs.items():
        try:
            run_name = _checked_name(run_name, "run name")
            raw = merge_configs(common, resolve_bases(fragment or {}, relative_to=path.parent))
            for override in overrides:
                apply_override(raw, override)
            # The run's key names the run, and every run writes into the suite's directory.
            raw["experiment"] = {
                **raw.get("experiment", {}),
                "run_name": run_name,
                "output_dir": str(suite_directory),
            }
            configs[run_name] = parse_config(raw)
        except ConfigError as error:
            problems.append(f"  {run_name}: {error}")
    if problems:
        raise ConfigError(f"{len(problems)} run(s) in {path} have config problems:\n" + "\n".join(problems))

    # Runs are only comparable if they are measured in exactly the same way.
    check_identical_measurement(configs)

    return Suite(name=suite_name, stages=stages, runs=configs, directory=suite_directory)


def run_suite(suite: Suite) -> dict[str, str]:
    """Check every run, then execute them in order and keep the suite report up to date.

    Returns:
        The runs that failed, mapped to a one-line description of the error.
        Empty if every run succeeded.

    Raises:
        PreflightError: If any run cannot start. Nothing is executed then.
    """
    _preflight(suite)

    failures: dict[str, str] = {}
    for position, (run_name, config) in enumerate(suite.runs.items(), start=1):
        logger.info("==== Run %d of %d: %s ====", position, len(suite.runs), run_name)
        try:
            _run_stages(config, suite.stages)
        except Exception as error:  # noqa: BLE001 - one broken run must not cost the others
            logger.exception("Run '%s' failed; continuing with the next run.", run_name)
            failures[run_name] = f"{type(error).__name__}: {error}"
        finally:
            _release_memory()
        write_report(suite.name, suite.directory, suite.runs, failures)

    return failures


def _preflight(suite: Suite) -> None:
    """Find every problem that would stop a run, before any run starts.

    Checks, for each run: that its datasets can be opened, that its view
    indices exist, that its model builds and satisfies the contract, and that
    an existing checkpoint was trained with the same model config.
    """
    problems = []
    for run_name, config in suite.runs.items():
        logger.info("Checking run '%s'", run_name)
        try:
            datasets = {split: build_dataset(config, split) for split in DATA_SPLITS}
            if "train" in suite.stages:
                check_training_settings(config, datasets["val"])
                _check_checkpoint_matches_model(config)
            check_evaluation_settings(config, datasets["test"])

            model = build_model(config).to(resolve_device(config.experiment.device))
            check_model_contract(model, image_size=dataset_image_size(datasets["test"]))
            del model
        except Exception as error:  # noqa: BLE001 - collect every problem, then report them together
            logger.debug("Preflight check of run '%s' failed", run_name, exc_info=True)
            problems.append(f"  {run_name}: {type(error).__name__}: {error}")
        finally:
            _release_memory()

    if problems:
        raise PreflightError(
            f"{len(problems)} of {len(suite.runs)} run(s) cannot start; nothing was run.\n" + "\n".join(problems)
        )


def _run_stages(config: Config, stages: list[str]) -> None:
    """Execute the requested stages of one run, skipping work that is already done.

    * train: skipped if the checkpoint has reached ``training.num_steps``;
      resumed if a checkpoint exists but has not; started otherwise.
    * evaluate: skipped if ``evaluation.json`` belongs to the current
      checkpoint and was made with the current evaluation settings.
    * profile: skipped if ``profile.json`` was made with the current settings.
    """
    files = RunFiles(config.run_dir)

    if "train" in stages:
        progress = read_progress(files.checkpoint)
        if progress is None:
            run_training(config)
        elif progress.step < config.training.num_steps:
            logger.info("Resuming training from step %d", progress.step)
            run_training(config, resume=True)
        else:
            logger.info("Training already finished (%d steps); skipping.", progress.step)

    if "evaluate" in stages:
        if _evaluation_is_current(config, files):
            logger.info("Evaluation is up to date; skipping.")
        else:
            run_evaluation(config)

    if "profile" in stages:
        if _output_settings_match(files.profile, profiling_settings_snapshot(config)):
            logger.info("Profile is up to date; skipping.")
        else:
            run_profiling(config)


def _evaluation_is_current(config: Config, files: RunFiles) -> bool:
    """Whether evaluation.json was produced from the current checkpoint with the current settings."""
    progress = read_progress(files.checkpoint)
    if progress is None or not _output_settings_match(files.evaluation, evaluation_settings_snapshot(config)):
        return False
    return json.loads(files.evaluation.read_text())["training_step"] == progress.step


def _output_settings_match(output_file: Path, settings: dict[str, Any]) -> bool:
    """Whether a JSON output file exists and records exactly these settings."""
    if not output_file.is_file():
        return False
    recorded = json.loads(output_file.read_text()).get("settings")
    # A round trip through JSON gives the current settings the same types
    # (lists, not tuples) as the recorded ones.
    return recorded == json.loads(json.dumps(settings))


def _check_checkpoint_matches_model(config: Config) -> None:
    """Refuse to resume a checkpoint that was trained with a different model config."""
    files = RunFiles(config.run_dir)
    if not files.checkpoint.exists() or not files.config.exists():
        return
    saved_model = yaml.safe_load(files.config.read_text()).get("model")
    if saved_model != config.raw["model"]:
        raise ConfigError(
            f"{files.checkpoint} was trained with a different model config. Give the run a new name, "
            f"or delete {config.run_dir} to train it from scratch."
        )


def _check_suite_keys(content: Any, path: Path) -> None:
    if not isinstance(content, Mapping):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level.")
    unknown = sorted(set(content) - _SUITE_KEYS)
    missing = sorted(_SUITE_KEYS - set(content))
    if unknown:
        raise ConfigError(f"Unknown key(s) in suite file {path}: {unknown}. Valid keys: {sorted(_SUITE_KEYS)}.")
    if missing:
        raise ConfigError(f"Missing key(s) in suite file {path}: {missing}.")


def _checked_name(name: Any, what: str) -> str:
    """Names become directory names, so they are restricted to safe characters."""
    if not isinstance(name, str) or not _VALID_NAME.match(name):
        raise ConfigError(
            f"Invalid {what} {name!r}: use only letters, digits and the characters _ . + -"
        )
    return name


def _checked_stages(stages: Any) -> list[str]:
    if not isinstance(stages, list) or not stages:
        raise ConfigError(f"'stages' must be a non-empty list chosen from {list(STAGES)}.")
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ConfigError(f"Unknown stage(s) {unknown}. Valid stages: {list(STAGES)}.")
    return [stage for stage in STAGES if stage in stages]


def _release_memory() -> None:
    """Free the memory of the previous run before the next one starts."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
