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
      hiera_nerf:
        base: ../models/hiera_nerf.yaml
        optimization:
          group_overrides: {encoder: {learning_rate: 1.0e-5}}

``common`` and each entry of ``runs`` are ordinary config fragments: they may
use ``base`` (paths relative to the suite file) and any config section. A
run's config is ``common``, then the run's own fragment, then any ``--set``
overrides from the command line. The run's key becomes its
``experiment.run_name``, and all runs write into
``<experiment.output_dir>/<suite_name>/``.

An optional ``seeds: [0, 1, 2]`` repeats every run once per seed, each in its
own directory ``<run>/seed_<seed>/`` with ``experiment.seed`` set to that
seed. For a given seed, all runs are still trained on identical data. The
report then shows each run's mean and spread over its seeds (see
``scene_recon.comparison``).

Running a suite:

1. Every run's config is built and validated. The runs must use identical
   measurement settings (see ``scene_recon.fairness``). Then everything each
   run's stages need is built and checked, without training: datasets,
   loss, metrics, optimiser, and the model against the pipeline's contract.
   A run that already has a checkpoint must ask for exactly the settings it
   was trained with (see ``scene_recon.provenance``). Any problem stops the
   suite here, before hours are spent on the first run.
2. The runs are executed one after another, each through the requested
   stages. Work that is still valid is skipped (see ``_run_stages``), so an
   interrupted suite is resumed by starting it again.
3. After every run the suite report (tables and plots) is rewritten, so
   partial results can be inspected while later runs are still training.

If a run fails, or the report cannot be written, the error is logged and the
suite moves on to the next run.
"""

from __future__ import annotations

import copy
import gc
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from scene_recon.checkpoints import read_progress
from scene_recon.config import Config, apply_override, merge_configs, parse_config, resolve_bases
from scene_recon.fairness import check_identical_measurement
from scene_recon.provenance import evaluation_status, profile_status
from scene_recon.registry import ConfigError
from scene_recon.report import write_report
from scene_recon.run_files import RunFiles, SuiteFiles
from scene_recon.stages import STAGES, check_run_can_start, run_evaluation, run_profiling, run_training

logger = logging.getLogger(__name__)

_SUITE_KEYS = {"suite_name", "stages", "common", "runs"}
_OPTIONAL_SUITE_KEYS = {"seeds"}
# Letters, digits and _ . + -, but not "." or ".." alone, which are not directory names of their own.
_VALID_NAME = re.compile(r"^(?!\.+$)[A-Za-z0-9_.+-]+$")


class PreflightError(Exception):
    """Raised when one or more runs of a suite cannot start. The message lists every problem found."""


class SuiteReportError(Exception):
    """Raised after all runs if the final suite report could not be written."""

    def __init__(self, message: str, failures: dict[str, str]) -> None:
        super().__init__(message)
        self.failures = failures
        """Runs that failed, as ``run_suite`` would have returned them."""


@dataclass(frozen=True)
class Suite:
    name: str
    stages: list[str]
    """The stages to execute, in execution order."""

    runs: dict[str, Config]
    """The config of everything that is executed, in execution order. Keyed by
    run name, or by ``<run name>/seed_<seed>`` when the suite lists ``seeds``."""

    directory: Path
    """Where the runs and the suite report are written: ``<output_dir>/<suite_name>/``."""

    groups: dict[str, list[str]]
    """Run name in the suite file -> its keys in ``runs`` (one per seed)."""

    def grouped_runs(self) -> dict[str, dict[str, Config]]:
        """Run name -> {key in ``runs``: config}, the shape the report expects."""
        return {name: {key: self.runs[key] for key in keys} for name, keys in self.groups.items()}


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
    seeds = _checked_seeds(content.get("seeds"))
    runs = content["runs"]
    if not isinstance(runs, Mapping) or not runs:
        raise ConfigError(f"'runs' in {path} must be a non-empty mapping from run name to config.")
    _check_run_names_are_distinct(list(runs))

    common = resolve_bases(content["common"] or {}, relative_to=path.parent)

    # Every run is checked before reporting, so that one attempt lists all mistakes.
    raw_configs, problems = {}, {}
    for run_name, fragment in runs.items():
        try:
            run_name = _checked_run_name(run_name)
            raw = merge_configs(common, resolve_bases(fragment or {}, relative_to=path.parent))
            for override in overrides:
                apply_override(raw, override)
            raw_configs[run_name] = raw
        except ConfigError as error:
            problems[run_name] = str(error)

    # The output directory is shared, so it can only be decided from the runs that merged.
    suite_directory = Path(_shared_output_dir(raw_configs)) / suite_name if raw_configs else Path(suite_name)
    configs: dict[str, Config] = {}
    groups: dict[str, list[str]] = {}
    for run_name, raw in raw_configs.items():
        groups[run_name] = []
        for key, seed in _replicates(run_name, seeds):
            replicate = copy.deepcopy(raw)
            # The key names the run, and every run writes into the suite's directory.
            replicate["experiment"] = {
                **replicate.get("experiment", {}),
                "run_name": key,
                "output_dir": str(suite_directory),
            }
            if seed is not None:
                replicate["experiment"]["seed"] = seed
            try:
                configs[key] = parse_config(replicate)
                groups[run_name].append(key)
            except ConfigError as error:
                problems[run_name] = str(error)

    if problems:
        # Listed in suite order, whichever check found the problem.
        listed = [f"  {name}: {problems[name]}" for name in runs if name in problems]
        raise ConfigError(f"{len(problems)} run(s) in {path} have config problems:\n" + "\n".join(listed))

    # Runs are only comparable if they are measured in exactly the same way.
    check_identical_measurement(configs)

    return Suite(name=suite_name, stages=stages, runs=configs, directory=suite_directory, groups=groups)


def run_suite(suite: Suite) -> dict[str, str]:
    """Check every run, then execute them in order and keep the suite report up to date.

    Returns:
        The runs that failed, mapped to a one-line description of the error.
        Empty if every run succeeded.

    Raises:
        PreflightError: If any run cannot start. Nothing is executed then.
        SuiteReportError: If, after all runs, the report could not be written.
            A report that fails after an earlier run does not stop later runs.
    """
    _preflight(suite)

    failures: dict[str, str] = {}
    report_error: str | None = None
    for position, (run_name, config) in enumerate(suite.runs.items(), start=1):
        logger.info("==== Run %d of %d: %s ====", position, len(suite.runs), run_name)
        try:
            _run_stages(config, suite.stages)
        except Exception as error:  # noqa: BLE001 - one broken run must not cost the others
            logger.exception("Run '%s' failed; continuing with the next run.", run_name)
            failures[run_name] = f"{type(error).__name__}: {error}"
        finally:
            _release_memory()

        try:
            write_report(suite.name, suite.directory, suite.grouped_runs(), failures, suite.stages)
            report_error = None
        except Exception as error:  # noqa: BLE001 - a broken report must not cost the remaining runs
            logger.exception("The suite report could not be written; continuing with the next run.")
            report_error = f"{type(error).__name__}: {error}"

    if report_error is not None:
        raise SuiteReportError(f"The suite report could not be written: {report_error}", failures)
    return failures


def _preflight(suite: Suite) -> None:
    """Find every problem that would stop a run, before any run starts."""
    problems = []
    for run_name, config in suite.runs.items():
        logger.info("Checking run '%s'", run_name)
        try:
            check_run_can_start(config, suite.stages)
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
    """Execute the requested stages of one run, skipping work that is still valid.

    * train: skipped if the checkpoint has reached ``training.num_steps``;
      resumed if a checkpoint exists but has not; started otherwise.
    * evaluate: skipped if ``evaluation.json`` was made from the current
      checkpoint file (same content) with the current measurement settings.
    * profile: skipped if ``profile.json`` was made with the current
      settings on the same kind of device with the same software.
    """
    if "train" in stages:
        progress = read_progress(RunFiles(config.run_dir).checkpoint)
        if progress is None:
            run_training(config)
        elif progress.step < config.training.num_steps:
            logger.info("Resuming training from step %d", progress.step)
            run_training(config, resume=True)
        else:
            logger.info("Training already finished (%d steps); skipping.", progress.step)

    if "evaluate" in stages:
        status = evaluation_status(config)
        if status.is_current:
            logger.info("Evaluation is up to date; skipping.")
        else:
            logger.info("Evaluating (%s).", status.reason)
            run_evaluation(config)

    if "profile" in stages:
        status = profile_status(config)
        if status.is_current:
            logger.info("Profile is up to date; skipping.")
        else:
            logger.info("Profiling (%s).", status.reason)
            run_profiling(config)


def _shared_output_dir(raw_configs: dict[str, dict[str, Any]]) -> str:
    """The output directory all runs agree on, after overrides."""
    output_dirs = {raw.get("experiment", {}).get("output_dir") for raw in raw_configs.values()}
    if None in output_dirs:
        raise ConfigError("Every run needs experiment.output_dir (usually set through base.yaml in 'common').")
    if len(output_dirs) > 1:
        raise ConfigError(
            f"All runs of a suite write into one directory, but they set different experiment.output_dir values "
            f"{sorted(output_dirs)}. Set it in 'common' (or with --set) rather than in individual runs."
        )
    return output_dirs.pop()


def _check_suite_keys(content: Any, path: Path) -> None:
    if not isinstance(content, Mapping):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level.")
    valid = _SUITE_KEYS | _OPTIONAL_SUITE_KEYS
    unknown = sorted(set(content) - valid)
    missing = sorted(_SUITE_KEYS - set(content))
    if unknown:
        raise ConfigError(f"Unknown key(s) in suite file {path}: {unknown}. Valid keys: {sorted(valid)}.")
    if missing:
        raise ConfigError(f"Missing key(s) in suite file {path}: {missing}.")


def _checked_name(name: Any, what: str) -> str:
    """Names become directory names, so they are restricted to safe characters."""
    if not isinstance(name, str) or not _VALID_NAME.match(name):
        raise ConfigError(
            f"Invalid {what} {name!r}: use letters, digits and the characters _ . + - "
            "(and not just dots)."
        )
    return name


def _checked_run_name(name: Any) -> str:
    """A run name also must not be the name of one of the suite's own outputs, in any upper/lower case."""
    name = _checked_name(name, "run name")
    if name.lower() in SuiteFiles.reserved_names():
        raise ConfigError(
            f"Run name {name!r} is reserved: the suite writes a file or directory of that name next to the "
            f"runs. Reserved names: {sorted(SuiteFiles.reserved_names())}."
        )
    return name


def _check_run_names_are_distinct(names: list[Any]) -> None:
    """Run names that differ only in upper/lower case share a directory on macOS and Windows."""
    seen: dict[str, Any] = {}
    for name in names:
        key = str(name).lower()
        if key in seen:
            raise ConfigError(
                f"Run names {seen[key]!r} and {name!r} differ only in upper/lower case; on some systems they "
                "would share a directory. Rename one of them."
            )
        seen[key] = name


def _checked_seeds(seeds: Any) -> list[int] | None:
    """``None`` (each run uses its own experiment.seed) or a non-empty list of distinct seeds."""
    if seeds is None:
        return None
    valid = (
        isinstance(seeds, list)
        and seeds
        and all(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds)
        and len(set(seeds)) == len(seeds)
    )
    if not valid:
        raise ConfigError(f"'seeds' must be a non-empty list of distinct non-negative whole numbers, got {seeds!r}.")
    return seeds


def _replicates(run_name: str, seeds: list[int] | None) -> list[tuple[str, int | None]]:
    """The keys a run is executed under: its name, or one ``<name>/seed_<seed>`` per seed."""
    if seeds is None:
        return [(run_name, None)]
    return [(f"{run_name}/seed_{seed}", seed) for seed in seeds]


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
