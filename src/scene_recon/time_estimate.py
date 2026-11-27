"""How long a suite will take on this machine, for a few step budgets.

``scene-recon estimate-time`` first times, on this machine, a training step of
every run and the scoring of one view with each of its evaluation protocols
(rendering and metrics, as the evaluation does them). It then multiplies
those times by what the suite will do: how many steps each run trains, how
often it is validated, and how many views each evaluation scores.
``scripts/run_comparison.sh`` runs it once, after its trial, so that another
step budget can be chosen before the real training starts.

The steps are timed at the start of training, in its first phase (see
``training.first_phase_steps``): with pixelNeRF's recipe, always two source
views, where the later steps take one or two. The estimate errs on the long side.

It also times the check of the training data that every checkpoint save
makes (``scene_recon.provenance``): on a slow volume, looking at each of
the split's files at every save can take hours in all.

With ``parallel_runs`` above 1 in the suite file, the runs that will train
at the same time are also timed at the same time, each in a process of its
own, so that they share the GPU as they will.

Left out: starting each run (building the model, loading its weights),
writing the checkpoint files, and any other work sharing the GPU.

``scripts/run_comparison.sh`` can also choose the step budget itself: the
largest that finishes within a given number of hours
(``largest_budget_within``).
"""

from __future__ import annotations

import gc
import json
import logging
import multiprocessing
import multiprocessing.connection
import multiprocessing.synchronize
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from scene_recon.build import build_dataset, build_metrics, configured_float32_matmul_precision, resolve_device
from scene_recon.config import Config, parse_config
from scene_recon.data import SceneDataset
from scene_recon.environment import device_description
from scene_recon.evaluation import Evaluator
from scene_recon.run_files import write_atomically
from scene_recon.suite import Suite
from scene_recon.training import Trainer

logger = logging.getLogger(__name__)

WARMUP_STEPS = 10
"""Training steps not timed: the first ones start the loading processes and let the GPU pick its kernels."""

TIMED_STEPS = 60
"""Training steps timed; each picks its number of source views at random, as in training."""

TIMED_SCENES = 2
TIMED_VIEWS_PER_SCENE = 5
"""Views scored and timed per evaluation protocol, after one untimed view."""

TOGETHER_TIMEOUT_SECONDS = 1800
"""How long runs timed at the same time wait for each other to be ready, before giving up."""


def estimate_time(
    suite: Suite, budgets: Sequence[int], first_scenes: int | None, first_target_views: int | None
) -> tuple[str, list[Estimate]]:
    """Time every run of the suite on this machine, then estimate the hours for each step budget.

    ``first_scenes`` and ``first_target_views`` say what the suite is scored
    on first, before it is scored as its evaluation settings say (``None``:
    as they say, at once). Returns a table of the estimates, as text, and the
    estimates.
    """
    speeds, runs_at_once = measure_speeds(suite)
    first_config = next(iter(suite.runs.values()))
    logger.info("Timing the check of the training data that every checkpoint save makes")
    saving = measure_saving(first_config)
    estimates = plan_hours(suite, speeds, saving, budgets, first_scenes, first_target_views, runs_at_once)
    device = device_description(resolve_device(first_config.experiment.device))["name"]
    text = format_estimates(speeds, saving, estimates, first_scenes, first_target_views, device, runs_at_once)
    return text, estimates


def largest_budget_within(estimates: Sequence[Estimate], hours: float) -> int | None:
    """The largest step budget whose first report is done within ``hours``; ``None`` if none is.

    The first report: every run trained, scored first (see ``estimate_time``)
    and profiled.
    """
    return max((estimate.steps for estimate in estimates if estimate.first_scores <= hours), default=None)


def save_estimates(estimates: Sequence[Estimate], path: Path) -> None:
    """Write the estimates as JSON, to be read back with ``load_estimates``."""
    content = json.dumps([asdict(estimate) for estimate in estimates], indent=2)
    write_atomically(path, lambda temporary_path: temporary_path.write_text(content, encoding="utf-8"))


def load_estimates(path: Path) -> list[Estimate]:
    """The estimates that ``save_estimates`` wrote."""
    return [Estimate(**estimate) for estimate in json.loads(path.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------- measuring


@dataclass(frozen=True)
class Speed:
    """How long one run's work takes on this machine."""

    seconds_per_step: float
    """One training step, including the wait for its batch."""

    seconds_per_view: dict[tuple[int, ...], float]
    """Scoring one view, by the source views of the evaluation protocol."""

    def scoring(self, source_views: Sequence[int]) -> float:
        """Seconds per view with these source views; the first protocol's for source views that were not timed."""
        return self.seconds_per_view.get(tuple(source_views), next(iter(self.seconds_per_view.values())))


def measure_speeds(suite: Suite) -> tuple[dict[str, Speed], int]:
    """Time every run as the suite will execute them. Returns the speeds and how many runs were timed at once.

    With ``parallel_runs`` above 1, the runs that will train at the same time
    are timed at the same time. If that fails, every run is timed alone, and
    the estimate counts them one after another: unless sharing the GPU is
    slower than taking turns on it, that errs on the long side.
    """
    if suite.parallel_runs > 1 and len(suite.runs) > 1:
        try:
            return _measure_together(suite), suite.parallel_runs
        except Exception:  # noqa: BLE001 - runs timed alone still give an estimate
            logger.exception("Timing the runs at the same time failed; timing each run alone instead.")
    speeds = {}
    for key, config in suite.runs.items():
        logger.info("Timing %s on this machine", key)
        speeds[key] = measure_speed(config)
    return speeds, 1


def _measure_together(suite: Suite) -> dict[str, Speed]:
    """Time the runs ``suite.parallel_runs`` at a time in suite order, each in a process of its own.

    The runs of a group start each timing (training, then scoring) together.
    Raises ``RuntimeError`` if any of them fails.
    """
    # A new interpreter for each run: CUDA cannot be used in a forked copy of a process that has used it.
    context = multiprocessing.get_context("spawn")
    runs = list(suite.runs.items())
    speeds = {}
    for start in range(0, len(runs), suite.parallel_runs):
        group = runs[start : start + suite.parallel_runs]
        logger.info("Timing %s at the same time on this machine", ", ".join(key for key, _ in group))
        together = context.Barrier(len(group))
        started = []
        try:
            for key, config in group:
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(target=_measure_in_own_process, args=(config, together, sender), name=key)
                process.start()
                sender.close()  # the run's process holds the only sending end now
                started.append((key, process, receiver))
            for key, process, receiver in started:
                try:
                    result = receiver.recv()
                except EOFError:  # the process ended before it could send its speed
                    result = f"its process stopped with exit code {process.exitcode}"
                if isinstance(result, str):
                    raise RuntimeError(f"Timing {key} failed: {result}")
                speeds[key] = result
        finally:
            together.abort()  # a run still waiting for the others stops waiting
            for _, process, receiver in started:
                process.join(timeout=60)
                if process.is_alive():
                    process.terminate()
                    process.join()
                receiver.close()
    return speeds


def _measure_in_own_process(
    config: Config,
    together: multiprocessing.synchronize.Barrier,
    results: multiprocessing.connection.Connection,
) -> None:
    """Time one run, in a process started by ``_measure_together``. Sends its ``Speed``, or the error as text."""
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s | {config.experiment.run_name} | %(message)s")
    try:
        speed = measure_speed(config, together)
    except Exception as error:  # noqa: BLE001 - reported to the suite's process, which times the runs alone
        together.abort()  # the other runs stop waiting for this one
        results.send(f"{type(error).__name__}: {error}")
    else:
        results.send(speed)
    finally:
        results.close()


def measure_speed(config: Config, together: multiprocessing.synchronize.Barrier | None = None) -> Speed:
    """Time a few training steps and scored views of a run, on its device.

    The run's own folder is never touched: the model is built afresh in a
    temporary folder, trained in memory and thrown away. ``together``, if
    given, is shared with the runs timed at the same time: each timing starts
    once all of them are ready for it.
    """
    with tempfile.TemporaryDirectory() as folder, configured_float32_matmul_precision(config):
        raw = config.raw
        training = dict(raw["training"])
        if training["first_phase_steps"] > 0:
            # Every timed step in the first phase, whose two source views cost the most.
            training["first_phase_steps"] = max(training["first_phase_steps"], WARMUP_STEPS + TIMED_STEPS)
        trainer = Trainer(
            parse_config({**raw, "experiment": {**raw["experiment"], "output_dir": folder}, "training": training})
        )
        try:
            _wait_for_the_others(together)
            seconds_per_step = trainer.seconds_per_step(WARMUP_STEPS, TIMED_STEPS)
            scenes = build_dataset(config, "test")
            metrics = build_metrics(config.evaluation.metrics)
            evaluator = Evaluator(trainer.model, trainer.renderer, metrics, trainer.device, config.experiment.seed)
            _wait_for_the_others(together)
            seconds_per_view = {
                tuple(source_views): _seconds_per_view(evaluator, scenes, source_views)
                for source_views in config.evaluation.source_view_sets
            }
        finally:
            trainer.writer.close()  # its files must be closed before the folder is deleted
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return Speed(seconds_per_step, seconds_per_view)


def _wait_for_the_others(together: multiprocessing.synchronize.Barrier | None) -> None:
    """Wait until every run timed at the same time is ready for the next timing (at once when timed alone)."""
    if together is not None:
        together.wait(timeout=TOGETHER_TIMEOUT_SECONDS)


@dataclass(frozen=True)
class Saving:
    """The check of the training data that every checkpoint save makes (its fingerprint)."""

    first_seconds: float
    """The first save in a process: reads and hashes every file of the training split."""

    seconds_per_save: float
    """Every later save: looks at the size and time of change of every file again."""

    test_first_seconds: float
    """The test split's first fingerprint in a process, which its evaluations and profiles record."""

    test_later_seconds: float
    """Every later fingerprint of the test split: looks at the size and time of change of every file again."""


def checks_of_test_data_per_command(num_runs: int) -> int:
    """How many times one suite command fingerprints the test split, at most.

    Each run's own stages check it up to 3 times, and the report, written
    again after every run, checks every run up to twice. Counted on the code;
    ``tests/test_time_estimate.py`` keeps the count honest.
    """
    return 3 * num_runs + 2 * num_runs**2


def measure_saving(config: Config) -> Saving:
    """Time the first and a later fingerprint of the run's training split, and of its test split."""
    seconds = []
    for split in ("train", "test"):
        scenes = build_dataset(config, split)
        for _ in range(2):
            start = time.perf_counter()
            scenes.fingerprint()
            seconds.append(time.perf_counter() - start)
    return Saving(*seconds)


def _seconds_per_view(evaluator: Evaluator, scenes: SceneDataset, source_views: Sequence[int]) -> float:
    evaluator.evaluate(scenes, source_views, num_scenes=1, num_target_views=1, num_preview_scenes=0)  # untimed
    _wait_for_gpu(evaluator.device)
    start = time.perf_counter()
    evaluator.evaluate(
        scenes, source_views, num_scenes=TIMED_SCENES, num_target_views=TIMED_VIEWS_PER_SCENE, num_preview_scenes=0
    )
    _wait_for_gpu(evaluator.device)
    return (time.perf_counter() - start) / _views(scenes, source_views, TIMED_SCENES, TIMED_VIEWS_PER_SCENE)


def _wait_for_gpu(device: torch.device) -> None:
    """Block until queued GPU work has finished, so that a timer stops after it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------- the estimate


@dataclass(frozen=True)
class Estimate:
    """Hours from the start of training until each milestone, for one step budget."""

    steps: int
    first_scores: float
    """Every run trained, profiled and scored on the first test scenes."""

    all_scores: float
    """Also scored on every test scene: the final report."""


class _Datasets:
    """Each split's scenes, loaded once however many runs read it."""

    def __init__(self) -> None:
        self._loaded: dict[tuple[str, str], SceneDataset] = {}

    def get(self, config: Config, split: str) -> SceneDataset:
        key = (json.dumps(config.raw["data"], sort_keys=True, default=str), split)
        if key not in self._loaded:
            self._loaded[key] = build_dataset(config, split)
        return self._loaded[key]


def plan_hours(
    suite: Suite,
    speeds: dict[str, Speed],
    saving: Saving,
    budgets: Sequence[int],
    first_scenes: int | None,
    first_target_views: int | None,
    runs_at_once: int,
) -> list[Estimate]:
    """The estimate for each step budget, from the speed of every run (by its key in the suite).

    Every run trains from step 0. Validation is timed as the evaluation
    protocol with the same source views, profiling as the first protocol.
    ``runs_at_once`` runs train and are scored at the same time, as they were
    timed; each group of them lasts as long as its slowest run. Profiling is
    counted one run at a time, as the suite does it. Each suite command
    (scored first, then as the evaluation settings say) is a process of its
    own, so each pays for the first fingerprint of the training split
    (checked before anything runs, also when training is finished) and of
    the test split, and checks the test split again (see
    ``checks_of_test_data_per_command``).
    """
    datasets = _Datasets()
    command = saving.first_seconds if "train" in suite.stages else 0.0
    if {"evaluate", "profile"} & set(suite.stages):
        later_checks = checks_of_test_data_per_command(len(suite.runs)) - 1
        command += saving.test_first_seconds + later_checks * saving.test_later_seconds
    scored_twice = (first_scenes is not None or first_target_views is not None) and "evaluate" in suite.stages
    together = [stage for stage in suite.stages if stage != "profile"]
    estimates = []
    for steps in budgets:
        first_runs, later_runs, profiling = [], [], 0.0
        for key, config in suite.runs.items():
            config = _with_steps(config, steps)
            speed = speeds[key]
            first_runs.append(_run_seconds(config, speed, saving, 0, together, first_scenes, first_target_views, datasets))
            if "profile" in suite.stages:
                profiling += _run_seconds(config, speed, saving, 0, ["profile"], None, None, datasets)
            if scored_twice:
                later_runs.append(_run_seconds(config, speed, saving, steps, ["evaluate"], None, None, datasets))
        first = command + _elapsed(first_runs, runs_at_once) + profiling
        later = command + _elapsed(later_runs, runs_at_once) if scored_twice else 0.0
        estimates.append(Estimate(steps, first / 3600, (first + later) / 3600))
    return estimates


def _elapsed(seconds: Sequence[float], runs_at_once: int) -> float:
    """How long runs that take ``seconds`` each take in all, started ``runs_at_once`` at a time in suite order.

    Each group lasts as long as its slowest run. The suite starts the next run
    as soon as any one ends, so this errs on the long side.
    """
    return sum(max(seconds[start : start + runs_at_once]) for start in range(0, len(seconds), runs_at_once))


def _with_steps(config: Config, steps: int) -> Config:
    raw = config.raw
    return parse_config({**raw, "training": {**raw["training"], "num_steps": steps}})


def _run_seconds(
    config: Config,
    speed: Speed,
    saving: Saving,
    start: int,
    stages: Sequence[str],
    num_scenes: int | None,
    num_target_views: int | None,
    datasets: _Datasets,
) -> float:
    """One run's stages, training from step ``start``.

    ``num_scenes`` and ``num_target_views``, if given, replace the evaluation's own.
    """
    seconds = 0.0
    if "train" in stages and config.training.num_steps > start:
        training, validation = config.training, config.validation
        seconds += (training.num_steps - start) * speed.seconds_per_step
        validations = training.num_steps // training.validate_every - start // training.validate_every
        scenes = datasets.get(config, "val")
        views = _views(scenes, validation.source_views, validation.num_scenes, validation.num_target_views)
        seconds += validations * views * speed.scoring(validation.source_views)
        # A save every checkpoint_every steps, and one at the end.
        saves = training.num_steps // training.checkpoint_every - start // training.checkpoint_every + 1
        seconds += saves * saving.seconds_per_save
    if "evaluate" in stages:
        evaluation = config.evaluation
        scenes = datasets.get(config, "test")
        count = evaluation.num_scenes if num_scenes is None else num_scenes
        per_scene = evaluation.num_target_views if num_target_views is None else num_target_views
        for source_views in evaluation.source_view_sets:
            seconds += _views(scenes, source_views, count, per_scene) * speed.scoring(source_views)
    if "profile" in stages:
        frames = config.profiling.num_frames + config.profiling.num_warmup_frames
        seconds += frames * speed.scoring(config.evaluation.source_view_sets[0])
    return seconds


def _views(scenes: SceneDataset, source_views: Sequence[int], num_scenes: int | None, per_scene: int | None) -> int:
    """How many views an evaluation scores; ``None`` means every scene, or every view that is not a source view."""
    count = len(scenes) if num_scenes is None else min(num_scenes, len(scenes))
    available = scenes.num_views(0) - len(source_views)
    return count * (available if per_scene is None else min(per_scene, available))


def format_estimates(
    speeds: dict[str, Speed],
    saving: Saving,
    estimates: Sequence[Estimate],
    first_scenes: int | None,
    first_target_views: int | None,
    device: str,
    runs_at_once: int,
) -> str:
    """The measured speeds and the estimates, as lines of plain text."""
    timed = "one run at a time" if runs_at_once == 1 else f"{runs_at_once} runs at the same time, as they will train"
    lines = [f"Measured on this machine ({device}), {timed}:"]
    for key, speed in speeds.items():
        scoring = ", ".join(
            f"{seconds:.2f} s with {len(views)} source view{'' if len(views) == 1 else 's'}"
            for views, seconds in speed.seconds_per_view.items()
        )
        lines.append(f"  {key}: {speed.seconds_per_step:.3f} s per training step; scoring a view {scoring}")
    lines.append(
        f"  checking the training data at each checkpoint save: {saving.seconds_per_save:.1f} s "
        f"({saving.first_seconds:.1f} s the first time in each command)"
    )
    lines.append(
        f"  checking the test data: {saving.test_first_seconds:.1f} s the first time in each command, "
        f"{saving.test_later_seconds:.1f} s each later time (up to {checks_of_test_data_per_command(len(speeds))} "
        "times in each command)"
    )
    scored_twice = first_scenes is not None or first_target_views is not None
    scenes = "all test scenes" if first_scenes is None else f"{first_scenes} test scenes"
    first = f"scored on {scenes}" if first_target_views is None else f"scored on {first_target_views} views of {scenes}"
    columns = ["steps per run", f"trained, {first}", "final report"]
    if not scored_twice:
        columns.pop()
    lines += ["", "Hours from the start of training until:", "  " + " | ".join(columns)]
    for estimate in estimates:
        values = [f"{estimate.steps:,}", f"{estimate.first_scores:.1f}"]
        if scored_twice:
            values.append(f"{estimate.all_scores:.1f}")
        lines.append("  " + " | ".join(value.rjust(len(column)) for value, column in zip(values, columns)))
    lines += [
        "",
        "Not counted: starting each run and writing the checkpoint files (minutes in all), and other work",
        "sharing the GPU.",
    ]
    return "\n".join(lines)
