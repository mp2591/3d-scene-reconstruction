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

An optional ``parallel_runs: 3`` trains and scores up to that many runs at
the same time, each in a process of its own on the same GPU. Their results
are the same as one after another. Rendering speed is not: it is measured
afterwards, one run at a time, with the GPU to itself. Without the key, runs
go one after another. With them in parallel, an optional
``report_every_minutes: 30`` also rewrites the report that often while they
train, with every run's curves so far.

Running a suite:

1. Every run's config is built and validated. The runs must use identical
   measurement settings (see ``scene_recon.fairness``). Then everything each
   run's stages need is built and checked, without training: datasets,
   loss, metrics, optimiser, and the model against the pipeline's contract.
   A run that already has a checkpoint must ask for exactly the settings it
   was trained with (see ``scene_recon.provenance``). Any problem stops the
   suite here, before hours are spent on the first run.
2. The runs are executed one after another (or ``parallel_runs`` at a
   time), each through the requested stages. Work that is still valid is
   skipped (see ``_run_stages``), so an interrupted suite is resumed by
   starting it again.
3. After every run the suite report (tables and plots) is rewritten, so
   partial results can be inspected while later runs are still training.

If a run fails, or the report cannot be written, the error is logged and the
suite moves on to the next run.
"""

from __future__ import annotations

import copy
import gc
import logging
import multiprocessing
import multiprocessing.connection
import os
import re
import threading
import time
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
_OPTIONAL_SUITE_KEYS = {"seeds", "parallel_runs", "report_every_minutes"}
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

    parallel_runs: int
    """How many runs train and are scored at the same time. 1: one after another."""

    report_every_minutes: float | None
    """While runs train in parallel, the report is also rewritten this often. ``None``: only after each run."""

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
    with open(path, encoding="utf-8") as file:
        content = yaml.safe_load(file) or {}

    _check_suite_keys(content, path)
    suite_name = _checked_name(content["suite_name"], "suite_name")
    stages = _checked_stages(content["stages"])
    seeds = _checked_seeds(content.get("seeds"))
    parallel_runs = _checked_parallel_runs(content.get("parallel_runs", 1))
    report_every_minutes = _checked_report_every_minutes(content.get("report_every_minutes"))
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

    return Suite(
        name=suite_name,
        stages=stages,
        runs=configs,
        directory=suite_directory,
        groups=groups,
        parallel_runs=parallel_runs,
        report_every_minutes=report_every_minutes,
    )


def run_suite(suite: Suite) -> dict[str, str]:
    """Check every run, then execute them and keep the suite report up to date.

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
    if suite.parallel_runs == 1:
        report_error = _run_one_after_another(suite, suite.stages, failures)
    else:
        # Training and scoring give the same results whatever else uses the
        # GPU, so several runs do them at once. Rendering speed does not: it
        # is measured afterwards, one run at a time, with the GPU to itself.
        report_error = None
        together = [stage for stage in suite.stages if stage != "profile"]
        if together:
            report_error = _run_in_parallel(suite, together, failures)
        if "profile" in suite.stages:
            report_error = _run_one_after_another(suite, ["profile"], failures)

    if report_error is not None:
        raise SuiteReportError(f"The suite report could not be written: {report_error}", failures)
    return failures


def _run_one_after_another(suite: Suite, stages: list[str], failures: dict[str, str]) -> str | None:
    """Execute ``stages`` of every run in this process, in suite order, rewriting the report after each.

    A run already in ``failures`` is skipped; a run that fails is added to it.
    Returns the error of the last report written, if it failed.
    """
    report_error = None
    for position, (run_name, config) in enumerate(suite.runs.items(), start=1):
        if run_name in failures:
            continue
        logger.info("==== Run %d of %d: %s ====", position, len(suite.runs), run_name)
        try:
            _run_stages(config, stages)
        except Exception as error:  # noqa: BLE001 - one broken run must not cost the others
            logger.exception("Run '%s' failed; continuing with the next run.", run_name)
            failures[run_name] = f"{type(error).__name__}: {error}"
        finally:
            _release_memory()
        report_error = _write_report(suite, failures)
    return report_error


def _run_in_parallel(suite: Suite, stages: list[str], failures: dict[str, str]) -> str | None:
    """Execute ``stages`` of every run, ``suite.parallel_runs`` at a time, each in a process of its own.

    The report is rewritten (by this process only) after each run ends, and
    every ``suite.report_every_minutes`` while runs go on. A run that fails
    is added to ``failures``. Returns the error of the last report written,
    if it failed.
    """
    # A new interpreter for each run: CUDA cannot be used in a forked copy of a process that has used it.
    context = multiprocessing.get_context("spawn")
    suite.directory.mkdir(parents=True, exist_ok=True)
    log_path = SuiteFiles(suite.directory).log
    waiting = list(suite.runs.items())
    running: dict[str, tuple[multiprocessing.Process, multiprocessing.connection.Connection]] = {}
    report_error = None
    interval = None if suite.report_every_minutes is None else suite.report_every_minutes * 60
    next_report = None if interval is None else time.monotonic() + interval
    try:
        while waiting or running:
            while waiting and len(running) < suite.parallel_runs:
                run_name, config = waiting.pop(0)
                logger.info(
                    "==== Starting %s (%s), up to %d runs at once ====", run_name, ", ".join(stages), suite.parallel_runs
                )
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_run_in_own_process, args=(config, stages, log_path, os.getpid(), sender), name=run_name
                )
                process.start()
                sender.close()  # the run's process holds the only sending end now
                running[run_name] = (process, receiver)

            timeout = None if next_report is None else max(0.0, next_report - time.monotonic())
            multiprocessing.connection.wait([process.sentinel for process, _ in running.values()], timeout=timeout)
            if next_report is not None and time.monotonic() >= next_report:
                report_error = _write_report(suite, failures)  # the runs' curves so far
                next_report = time.monotonic() + interval
            for run_name, (process, receiver) in list(running.items()):
                if process.is_alive():
                    continue
                process.join()
                try:
                    error = receiver.recv()
                except EOFError:  # the process ended before it could say how it went
                    error = f"its process stopped with exit code {process.exitcode}"
                receiver.close()
                del running[run_name]
                if error is None:
                    logger.info("Run '%s' finished.", run_name)
                else:
                    logger.error("Run '%s' failed: %s", run_name, error)
                    failures[run_name] = error
                report_error = _write_report(suite, failures)
    finally:
        for process, _ in running.values():  # only if this process was stopped, e.g. by Ctrl-c
            process.terminate()
            process.join()
    return report_error


def _run_in_own_process(
    config: Config,
    stages: list[str],
    log_path: Path,
    parent_pid: int,
    results: multiprocessing.connection.Connection,
) -> None:
    """The work of one run, in a process started by ``_run_in_parallel``.

    Its log lines go to the terminal and to the suite's log, marked with the
    run's name. It sends ``None`` back when it succeeds, or a one-line
    description of the error.
    """
    _stop_with_parent(parent_pid)
    run_name = config.experiment.run_name
    log_format = logging.Formatter(f"%(asctime)s | {run_name} | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path)):
        handler.setFormatter(log_format)
        root.addHandler(handler)
    try:
        _run_stages(config, stages)
    except Exception as error:  # noqa: BLE001 - reported to the suite, which carries on with the other runs
        logger.exception("Run '%s' failed.", run_name)
        results.send(f"{type(error).__name__}: {error}")
    else:
        results.send(None)
    finally:
        results.close()


def _stop_with_parent(parent_pid: int) -> None:
    """End this process within seconds once the process that started it has ended.

    Otherwise a run would go on training after the suite was stopped (for
    example with pkill), and starting the suite again would train it twice
    at once. On Linux an orphaned process gets another parent, which is what
    is watched for.
    """

    def watch() -> None:
        while os.getppid() == parent_pid:
            time.sleep(5)
        os._exit(1)

    threading.Thread(target=watch, name="stop-with-parent", daemon=True).start()


def _write_report(suite: Suite, failures: dict[str, str]) -> str | None:
    """Rewrite the suite report. Returns ``None``, or the error if it could not be written."""
    try:
        write_report(suite.name, suite.directory, suite.grouped_runs(), failures, suite.stages)
    except Exception as error:  # noqa: BLE001 - a broken report must not cost the remaining runs
        logger.exception("The suite report could not be written; carrying on.")
        return f"{type(error).__name__}: {error}"
    return None


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


def _checked_report_every_minutes(minutes: Any) -> float | None:
    """How often the report is rewritten while runs train in parallel: a positive number of minutes, or unset."""
    if minutes is None:
        return None
    if not isinstance(minutes, (int, float)) or isinstance(minutes, bool) or not minutes > 0:
        raise ConfigError(f"'report_every_minutes' must be a positive number of minutes, got {minutes!r}.")
    return float(minutes)


def _checked_parallel_runs(parallel_runs: Any) -> int:
    """How many runs train and are scored at once: a whole number of at least 1."""
    if not isinstance(parallel_runs, int) or isinstance(parallel_runs, bool) or parallel_runs < 1:
        raise ConfigError(f"'parallel_runs' must be a whole number of at least 1, got {parallel_runs!r}.")
    return parallel_runs


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
