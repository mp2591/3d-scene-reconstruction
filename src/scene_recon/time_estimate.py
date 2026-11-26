"""How long the pilot and the comparison will take on this machine, for a few step budgets.

``scene-recon estimate-time`` first times, on this machine, a training step of
every comparison run and the scoring of one view with each of its
evaluation protocols (rendering and metrics, as the evaluation does them).
It then multiplies those times by what the pilot and the comparison will
do: how many steps each run trains, how often it is validated, and how many
views each evaluation scores. ``scripts/run_comparison.sh`` runs it once,
after its trial, so that a larger step budget can be chosen before the
pilot starts.

Left out: starting each run (building the model, loading its weights), the
first fingerprint of the training data, saving checkpoints, choosing the
rates, and any other work sharing the GPU.
"""

from __future__ import annotations

import gc
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from scene_recon.build import build_dataset, build_metrics, configured_float32_matmul_precision, resolve_device
from scene_recon.config import Config, parse_config
from scene_recon.data import SceneDataset
from scene_recon.environment import device_description
from scene_recon.evaluation import Evaluator
from scene_recon.pilot import pair_pilot_with_comparison
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


def estimate_time(pilot: Suite, comparison: Suite, budgets: Sequence[int], first_scenes: int | None) -> str:
    """Time every comparison run on this machine, then estimate the hours for each step budget, as a table.

    ``first_scenes`` is the number of test scenes the comparison is scored on
    first, before all of them (``None``: all of them at once).
    """
    speeds = {}
    for key, config in comparison.runs.items():
        logger.info("Timing %s on this machine", key)
        speeds[key] = measure_speed(config)
    estimates = plan_hours(pilot, comparison, speeds, budgets, first_scenes)
    device = resolve_device(next(iter(comparison.runs.values())).experiment.device)
    return format_estimates(speeds, estimates, first_scenes, device_description(device)["name"])


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


def measure_speed(config: Config) -> Speed:
    """Time a few training steps and scored views of a run, on its device.

    The run's own folder is never touched: the model is built afresh in a
    temporary folder, trained in memory and thrown away.
    """
    with tempfile.TemporaryDirectory() as folder, configured_float32_matmul_precision(config):
        raw = config.raw
        trainer = Trainer(parse_config({**raw, "experiment": {**raw["experiment"], "output_dir": folder}}))
        try:
            seconds_per_step = trainer.seconds_per_step(WARMUP_STEPS, TIMED_STEPS)
            scenes = build_dataset(config, "test")
            metrics = build_metrics(config.evaluation.metrics)
            evaluator = Evaluator(trainer.model, trainer.renderer, metrics, trainer.device)
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
    """Hours from the start of the pilot until each milestone, for one step budget."""

    steps: int
    pilot_done: float
    first_scores: float
    """The comparison trained, profiled and scored on the first test scenes."""

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
    pilot: Suite,
    comparison: Suite,
    speeds: dict[str, Speed],
    budgets: Sequence[int],
    first_scenes: int | None,
) -> list[Estimate]:
    """The estimate for each step budget, from the speed of every comparison run (by its key in the suite).

    Each pilot run is timed as the comparison run it trains like. A tuned
    comparison run continues from the pilot's last step; any other starts
    from step 0. Validation is timed as the evaluation protocol with the same
    source views, profiling as the first protocol.
    """
    datasets = _Datasets()
    tuned = pair_pilot_with_comparison(pilot, comparison)
    starts = {run.key: run.candidates[0].config.training.num_steps for run in tuned}

    pilot_seconds = 0.0
    for run in tuned:
        for candidate in run.candidates:
            pilot_seconds += _run_seconds(candidate.config, speeds[run.key], 0, pilot.stages, None, datasets)

    estimates = []
    for steps in budgets:
        first = later = 0.0
        for key, config in comparison.runs.items():
            config = _with_steps(config, steps)
            first += _run_seconds(config, speeds[key], starts.get(key, 0), comparison.stages, first_scenes, datasets)
            if first_scenes is not None and "evaluate" in comparison.stages:
                later += _run_seconds(config, speeds[key], steps, ["evaluate"], None, datasets)
        pilot_hours, first_hours, later_hours = (seconds / 3600 for seconds in (pilot_seconds, first, later))
        estimates.append(
            Estimate(steps, pilot_hours, pilot_hours + first_hours, pilot_hours + first_hours + later_hours)
        )
    return estimates


def _with_steps(config: Config, steps: int) -> Config:
    raw = config.raw
    return parse_config({**raw, "training": {**raw["training"], "num_steps": steps}})


def _run_seconds(
    config: Config,
    speed: Speed,
    start: int,
    stages: Sequence[str],
    num_scenes: int | None,
    datasets: _Datasets,
) -> float:
    """One run's stages, training from step ``start``; ``num_scenes``, if given, replaces evaluation.num_scenes."""
    seconds = 0.0
    if "train" in stages and config.training.num_steps > start:
        training, validation = config.training, config.validation
        seconds += (training.num_steps - start) * speed.seconds_per_step
        validations = training.num_steps // training.validate_every - start // training.validate_every
        scenes = datasets.get(config, "val")
        views = _views(scenes, validation.source_views, validation.num_scenes, validation.num_target_views)
        seconds += validations * views * speed.scoring(validation.source_views)
    if "evaluate" in stages:
        evaluation = config.evaluation
        scenes = datasets.get(config, "test")
        for source_views in evaluation.source_view_sets:
            count = evaluation.num_scenes if num_scenes is None else num_scenes
            seconds += _views(scenes, source_views, count, evaluation.num_target_views) * speed.scoring(source_views)
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
    speeds: dict[str, Speed], estimates: Sequence[Estimate], first_scenes: int | None, device: str
) -> str:
    """The measured speeds and the estimates, as lines of plain text."""
    lines = [f"Measured on this machine ({device}):"]
    for key, speed in speeds.items():
        scoring = ", ".join(
            f"{seconds:.2f} s with {len(views)} source view{'' if len(views) == 1 else 's'}"
            for views, seconds in speed.seconds_per_view.items()
        )
        lines.append(f"  {key}: {speed.seconds_per_step:.3f} s per training step; scoring a view {scoring}")
    first = "scored on all test scenes" if first_scenes is None else f"scored on {first_scenes} test scenes"
    columns = ["steps per run", "pilot done", f"trained, {first}", "final report"]
    if first_scenes is None:
        columns.pop()
    lines += ["", "Hours from the start of the pilot until:", "  " + " | ".join(columns)]
    for estimate in estimates:
        values = [f"{estimate.steps:,}", f"{estimate.pilot_done:.1f}", f"{estimate.first_scores:.1f}"]
        if first_scenes is not None:
            values.append(f"{estimate.all_scores:.1f}")
        lines.append("  " + " | ".join(value.rjust(len(column)) for value, column in zip(values, columns)))
    lines += [
        "",
        "Not counted: starting each run, the first fingerprint of the training data, saving checkpoints,",
        "choosing the rates (minutes in all), and other work sharing the GPU.",
    ]
    return "\n".join(lines)
