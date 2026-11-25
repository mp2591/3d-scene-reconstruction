"""The suite report: tables and figures comparing every run of a suite.

Written into the suite directory and rewritten after every run, so partial
results can be inspected while later runs are still training::

    summary.md                                status, cost and quality tables, comparisons, figures
    summary.csv                               one row per run and seed, for your own analysis
    paired_comparison.csv                     per-scene differences to the reference run, with intervals
    plots/metrics_vs_source_views.png         each metric against the number of input views
    plots/quality_vs_speed.png                quality against rendering speed
    plots/training_curves.png                 training loss and validation quality over training
    plots/renders_<n>_source_views.png        the same views rendered by every run, next to the truth

A "run" is an entry of the suite file. When the suite lists several
``seeds``, each run was executed once per seed, and the report shows the
mean and standard deviation over seeds (see ``scene_recon.comparison``).

The report only reads the files each run wrote (see ``scene_recon.run_files``),
so it can be regenerated at any time without touching the models. It shows
a run's evaluation or profile only if it still matches the run's current
checkpoint and settings (see ``scene_recon.provenance``); out-of-date or
damaged results are left out of every table and figure, and listed with the
reason instead.

A current result is also only *eligible* for the requested experiment if the
suite requested no training, or its checkpoint has reached the requested
number of training steps. A result that is current but not eligible, e.g.
the evaluation of the old checkpoint after a training extension failed, is
*interim*: the quality tables still show it, marked, with its actual step,
but it is left out of best values, figures and paired comparisons.

A saved evaluation stays current when the software changes, so that the
report can be made on another machine. Every evaluation saves what it was
made with: Python, CUDA, the device's type and model, and every installed
package with its version (see ``environment.evaluation_software``, which
also lists what the record leaves out). The report compares these records
whole: if two runs' records differ, or one cannot be compared, it says so,
lists what differs or why, and gives no plain verdict in their paired
comparison. It cannot tell which package a score depends on, so it may
also flag a difference that does not matter; the list lets the reader judge.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import scene_recon.metrics  # noqa: F401  (registers the metrics, whose display settings are read below)
from scene_recon.checkpoints import read_checkpoint_starting_weights, read_checkpoint_training_data, read_progress
from scene_recon.comparison import PairedDifference, Summary, average_over_seeds, paired_difference, summarize
from scene_recon.config import Config
from scene_recon.environment import is_valid_package
from scene_recon.evaluation import load_preview_images
from scene_recon.fairness import MISSING, differing_settings, format_value, trained_on_identical_data
from scene_recon.plots import (
    plot_image_grid,
    plot_metrics_vs_source_views,
    plot_quality_vs_speed,
    plot_training_curves,
    run_styles,
)
from scene_recon.provenance import evaluation_status, profile_status
from scene_recon.registry import METRICS, ConfigError
from scene_recon.run_files import RunFiles, SuiteFiles, write_atomically

logger = logging.getLogger(__name__)

# The metric used for the quality-versus-speed figure and the validation
# curves, if the runs report it; otherwise the first metric is used.
_HEADLINE_METRIC = "psnr"


@dataclass
class RunResult:
    """What one executed run (one seed of a suite entry) produced, read from its output files."""

    key: str
    """The run's key in the suite, e.g. ``"wide"`` or ``"wide/seed_1"``."""

    seed: int
    status: str
    files: RunFiles
    training_step: int | None = None
    training_hours: float | None = None
    parameters: int | None = None
    frames_per_second: float | None = None

    metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    """Number of source views -> metric name -> mean over the test scenes."""

    per_scene: dict[int, dict[str, dict[str, float]]] = field(default_factory=dict)
    """Number of source views -> scene id -> metric name -> value."""

    preview_scene_ids: dict[int, list[str]] = field(default_factory=dict)
    """Number of source views -> scenes with a saved preview render."""

    left_out: list[str] = field(default_factory=list)
    """Saved results that are not shown because they are out of date or damaged, with the reason."""

    training_data: dict | None = None
    """The training-data fingerprint recorded in the run's checkpoint, if it has one."""

    starting_weights: dict | None = None
    """The pretrained weights recorded in the run's checkpoint (see ``pretrained_weights_of``), if it has one."""

    evaluation_software: dict | None = None
    """What its current evaluation was made with (see ``environment.evaluation_software``), checked.

    ``None`` if it has no current evaluation, or its record cannot be compared (see ``software_problems``).
    """

    software_problems: list[str] = field(default_factory=list)
    """Why the record of its current evaluation cannot be compared: keys of ``SOFTWARE_PROBLEMS``."""

    requested_steps: int = 0
    """``training.num_steps`` of the run's config."""

    eligible: bool = False
    """Whether its results count for the requested experiment (see the module docstring)."""

    interim_reason: str = ""
    """Why current results are not eligible, if they are not."""


@dataclass
class GroupResult:
    """One entry of the suite file, over all its seeds."""

    name: str
    replicates: list[RunResult]

    @property
    def with_results(self) -> list[RunResult]:
        """Seeds with a current evaluation, eligible or interim."""
        return [replicate for replicate in self.replicates if replicate.metrics]

    @property
    def eligible(self) -> list[RunResult]:
        """Seeds whose results count for the requested experiment."""
        return [replicate for replicate in self.with_results if replicate.eligible]

    @property
    def interim(self) -> list[RunResult]:
        """Seeds with current results that do not (yet) count for the requested experiment."""
        return [replicate for replicate in self.with_results if not replicate.eligible]

    def summary(self, count: int, metric: str, interim: bool = False) -> Summary | None:
        """Mean and spread over seeds of one metric for one number of input views.

        Uses the eligible seeds, or with ``interim=True`` the interim ones.
        """
        runs = self.interim if interim else self.eligible
        return summarize([run.metrics[count][metric] for run in runs if metric in run.metrics.get(count, {})])

    def scene_scores(self, count: int, metric: str, seeds: set[int]) -> dict[str, float]:
        """Per-scene values of one metric, averaged over the given seeds."""
        return average_over_seeds(
            [
                {scene: values[metric] for scene, values in run.per_scene[count].items() if metric in values}
                for run in self.eligible
                if run.seed in seeds and count in run.per_scene
            ]
        )

    def seeds_with_result(self, count: int, metric: str) -> set[int]:
        """The eligible seeds whose evaluation has per-scene values of this metric for this number of views."""
        return {
            run.seed
            for run in self.eligible
            if any(metric in values for values in run.per_scene.get(count, {}).values())
        }

    @property
    def seeds(self) -> list[int]:
        """Every seed this run is meant to be trained with."""
        return [run.seed for run in self.replicates]

    @property
    def status(self) -> str:
        if len(self.replicates) == 1:
            return self.replicates[0].status
        counts: dict[str, int] = {}
        for replicate in self.replicates:
            state = replicate.status.split(" (")[0]
            counts[state] = counts.get(state, 0) + 1
        return ", ".join(f"{count} of {len(self.replicates)} {state}" for state, count in counts.items())

    @property
    def training_step(self) -> int | None:
        steps = [run.training_step for run in self.replicates if run.training_step is not None]
        return min(steps) if steps else None

    @property
    def training_steps_text(self) -> str:
        """The saved step count, and the requested one if training has not reached it."""
        step = self.training_step
        if step is None:
            return "–"
        requested = max(run.requested_steps for run in self.replicates)
        return f"{step:,}" if step >= requested else f"{step:,} of {requested:,}"

    @property
    def training_hours(self) -> float | None:
        return _mean_or_none([run.training_hours for run in self.replicates])

    @property
    def parameters(self) -> int | None:
        return next((run.parameters for run in self.replicates if run.parameters is not None), None)

    @property
    def frames_per_second(self) -> float | None:
        return _mean_or_none([run.frames_per_second for run in self.replicates])


def write_report(
    suite_name: str,
    directory: Path,
    groups: Mapping[str, Mapping[str, Config]],
    failures: Mapping[str, str],
    stages: Sequence[str],
) -> None:
    """Write the tables and figures of a suite.

    Args:
        suite_name: Shown as the report's title.
        directory: The suite directory.
        groups: Every run of the suite file, in suite order, mapped to its
            executed runs (one per seed): ``{run name: {key: config}}``.
        failures: Executed runs that failed, by key, mapped to their error message.
        stages: The stages the suite was asked to run. If they include
            "train", results only count once training reached its requested
            number of steps.
    """
    training_requested = "train" in stages
    results = [
        GroupResult(
            name,
            [_collect(key, config, key in failures, training_requested) for key, config in runs.items()],
        )
        for name, runs in groups.items()
    ]
    first_configs = {name: next(iter(runs.values())) for name, runs in groups.items()}
    seeds = [config.experiment.seed for config in next(iter(groups.values())).values()]
    comparisons = _paired_comparisons(results)

    files = SuiteFiles(directory)
    plots_dir = files.plots
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results, files.summary_csv)
    _write_comparison_csv(comparisons, files.paired_comparison_csv)
    figures = _write_figures(results, plots_dir, multiple_seeds=len(seeds) > 1)
    _write_markdown(
        suite_name,
        results,
        seeds,
        comparisons,
        figures,
        failures,
        differing_settings(first_configs),
        files.summary_markdown,
    )
    logger.info("Suite report updated: %s", files.summary_markdown)


SOFTWARE_SAME, SOFTWARE_DIFFERENT, SOFTWARE_UNKNOWN = "same", "different", "unknown"
"""Whether evaluations were made with the same software (see ``environment.evaluation_software``):
identical records, different records, or at least one record missing or damaged."""


def _compare_software(runs: list[RunResult]) -> str:
    """``SOFTWARE_SAME``, ``SOFTWARE_DIFFERENT`` or ``SOFTWARE_UNKNOWN`` for the evaluations of these runs."""
    records = [run.evaluation_software for run in runs]
    if any(record is None for record in records):
        return SOFTWARE_UNKNOWN
    return SOFTWARE_SAME if all(record == records[0] for record in records) else SOFTWARE_DIFFERENT


SOFTWARE_PROBLEMS = {
    "not recorded": "made before evaluations recorded their software; evaluate it again (`scene-recon evaluate`)",
    "device model unknown": (
        "the model of the device it ran on cannot be named (only NVIDIA GPUs, and CPUs whose model the "
        "operating system reports, are named)"
    ),
    "package list unreadable": (
        "the metadata of an installed package could not be read; the evaluation's log names the package, "
        "and reinstalling it fixes this"
    ),
    "damaged": "its record in evaluation.json is damaged or from an older version; evaluate it again",
}
"""Why an evaluation's software cannot be compared, and what to do about it."""

_RUNTIME_LABELS = {"python": "Python", "cuda": "CUDA", "device_type": "Device type", "device": "Device"}
"""A record's fields besides its packages, with capitalised labels, which a package name (lower case) cannot take."""

_RUNTIME_FIELDS = tuple(_RUNTIME_LABELS)


def _read_software(record) -> tuple[dict | None, list[str]]:
    """An evaluation's saved software, checked, or ``None`` and every reason it cannot be compared.

    The reasons are keys of ``SOFTWARE_PROBLEMS``, in its order; the list is empty for a readable record.
    """
    if record is None:
        return None, ["not recorded"]
    if not isinstance(record, dict) or set(record) != {*_RUNTIME_FIELDS, "packages"}:
        return None, ["damaged"]
    problems = set()
    device, packages = record["device"], record["packages"]
    if device is None:
        problems.add("device model unknown")
    if packages is None:
        problems.add("package list unreadable")
    runtime_valid = all(_is_text(record[name]) for name in ("python", "cuda", "device_type")) and (
        device is None or _is_text(device)  # None has its own reason, above
    )
    packages_valid = packages is None or (
        isinstance(packages, dict) and bool(packages) and all(is_valid_package(n, v) for n, v in packages.items())
    )
    if not runtime_valid or not packages_valid:
        problems.add("damaged")
    if problems:
        return None, [problem for problem in SOFTWARE_PROBLEMS if problem in problems]
    return {**{name: record[name] for name in _RUNTIME_FIELDS}, "packages": dict(sorted(packages.items()))}, []


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value)


def _software_for_display(record: dict) -> dict[str, str]:
    """A checked record as one label -> version mapping; a package called ``device`` stays apart from ``Device``."""
    return {**{label: record[name] for name, label in _RUNTIME_LABELS.items()}, **record["packages"]}


def _collect(key: str, config: Config, failed: bool, training_requested: bool) -> RunResult:
    """Read what a run has produced, keeping only results that still describe the run."""
    files = RunFiles(config.run_dir)
    result = RunResult(
        key=key,
        seed=config.experiment.seed,
        status="not started",
        files=files,
        requested_steps=config.training.num_steps,
    )

    progress = read_progress(files.checkpoint)
    finished_training = progress is not None and progress.step >= config.training.num_steps
    if progress is not None:
        result.training_data = read_checkpoint_training_data(files.checkpoint)
        result.starting_weights = read_checkpoint_starting_weights(files.checkpoint)
        result.training_step = progress.step
        result.training_hours = progress.training_seconds / 3600
        result.status = (
            "trained" if finished_training else f"training (step {progress.step:,} of {config.training.num_steps:,})"
        )

    evaluation = evaluation_status(config)
    if evaluation.is_current:
        result.evaluation_software, result.software_problems = _read_software(evaluation.content.get("software"))
        for protocol in evaluation.content["protocols"]:
            count = len(protocol["source_views"])
            result.metrics[count] = protocol["mean"]
            result.per_scene[count] = protocol.get("per_scene", {})
            result.preview_scene_ids[count] = protocol.get("preview_scene_ids", [])
        if progress is None:  # a model without trainable parameters, evaluated as built
            result.status = "evaluated (model as built)"
            result.eligible = True
        elif finished_training or not training_requested:
            result.status = "evaluated" if finished_training else f"evaluated at step {progress.step:,}"
            result.eligible = True
        else:
            result.status = f"evaluated at step {progress.step:,}"
            result.interim_reason = (
                f"evaluated at step {progress.step:,}, but {config.training.num_steps:,} training steps were "
                "requested" + (" and training failed" if failed else " and have not been reached yet")
            )
    elif evaluation.state != "missing":
        result.left_out.append(f"evaluation: {evaluation.reason}")

    profile = profile_status(config)
    if profile.is_current:
        result.parameters = profile.content["parameters"]["total"]
        result.frames_per_second = profile.content["rendering"]["frames_per_second"]
    elif profile.state != "missing":
        result.left_out.append(f"profile: {profile.reason}")

    if failed:
        result.status = "failed"
    return result


# ---------------------------------------------------------------- comparisons


@dataclass(frozen=True)
class _Comparison:
    run: str
    reference: str
    count: int
    metric: str
    difference: PairedDifference

    seeds: list[int]
    """The seeds compared: those both runs have results for."""

    partial: bool
    """True if some of the suite's seeds are not (yet) available on both sides."""

    software: str
    """Whether the compared evaluations were made with the same software (``SOFTWARE_SAME``, ...)."""

    @property
    def verdict(self) -> str:
        """better, worse or no clear difference; qualified unless the scores came from the same software."""
        verdict = self.difference.verdict(_metric_presentation(self.metric)[1])
        if self.software == SOFTWARE_DIFFERENT:
            return f"{verdict}, but scored with different software"
        if self.software == SOFTWARE_UNKNOWN:
            return f"{verdict}, but the software of a score is unknown"
        return verdict


def _paired_comparisons(results: list[GroupResult]) -> list[_Comparison]:
    """Every run against the reference, the first run of the suite file, seed by seed.

    Only seeds that both runs have results for are compared, so a seed that
    has finished for one run but not yet for the other cannot shift the
    difference. If the reference has no results yet, nothing is compared; the
    report says so instead of silently picking another reference.
    """
    if len(results) < 2 or not results[0].eligible:
        return []
    reference, others = results[0], results[1:]
    comparisons = []
    for count in _view_counts(results):
        for metric in _metric_names(results):
            reference_seeds = reference.seeds_with_result(count, metric)
            for group in others:
                seeds = reference_seeds & group.seeds_with_result(count, metric)
                if not seeds:
                    continue
                difference = paired_difference(
                    group.scene_scores(count, metric, seeds), reference.scene_scores(count, metric, seeds)
                )
                if difference is None:
                    continue
                partial = seeds != set(reference.seeds) or seeds != set(group.seeds)
                compared_runs = [run for side in (reference, group) for run in side.replicates if run.seed in seeds]
                comparisons.append(
                    _Comparison(
                        group.name,
                        reference.name,
                        count,
                        metric,
                        difference,
                        sorted(seeds),
                        partial,
                        _compare_software(compared_runs),
                    )
                )
    return comparisons


# ---------------------------------------------------------------- tables


def _metric_names(results: list[GroupResult]) -> list[str]:
    """Every metric reported by any run, in the order the runs list them."""
    names: list[str] = []
    for group in results:
        for run in group.with_results:
            for per_metric in run.metrics.values():
                names.extend(name for name in per_metric if name not in names)
    return names


def _view_counts(results: list[GroupResult]) -> list[int]:
    return sorted({count for group in results for run in group.with_results for count in run.metrics})


def _write_csv(results: list[GroupResult], path: Path) -> None:
    """One row per run and seed. Quality columns are named ``<metric>_<n>v``, e.g. ``psnr_1v``."""
    metric_names, view_counts = _metric_names(results), _view_counts(results)
    quality_columns = [f"{metric}_{count}v" for count in view_counts for metric in metric_names]
    columns = ["run", "seed", "status", "training_steps", "training_hours", "parameters", "frames_per_second"]
    software_column = "evaluation_software"  # a short hash of the record: equal hashes, equal software

    file = io.StringIO()
    writer = csv.DictWriter(file, fieldnames=columns + quality_columns + [software_column])
    writer.writeheader()
    for group in results:
        for run in group.replicates:
            row = {
                "run": group.name,
                "seed": run.seed,
                "status": run.status,
                "training_steps": run.training_step,
                "training_hours": run.training_hours,
                "parameters": run.parameters,
                "frames_per_second": run.frames_per_second,
            }
            for count, per_metric in run.metrics.items():
                for metric, value in per_metric.items():
                    row[f"{metric}_{count}v"] = value
            if run.evaluation_software is not None:
                row[software_column] = _short_hash(run.evaluation_software)
            writer.writerow(row)
    # The csv module already ends rows with "\r\n"; newline="" stops Windows adding another "\r".
    write_atomically(path, lambda temporary_path: temporary_path.write_text(file.getvalue(), encoding="utf-8", newline=""))


def _write_comparison_csv(comparisons: list[_Comparison], path: Path) -> None:
    columns = [
        "run",
        "reference",
        "num_source_views",
        "metric",
        "mean_difference",
        "ci95_low",
        "ci95_high",
        "num_scenes",
        "seeds",
        "partial",
        "software",
        "verdict",
    ]
    file = io.StringIO()
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    for comparison in comparisons:
        difference = comparison.difference
        writer.writerow(
            {
                "run": comparison.run,
                "reference": comparison.reference,
                "num_source_views": comparison.count,
                "metric": comparison.metric,
                "mean_difference": difference.mean,
                "ci95_low": difference.low,
                "ci95_high": difference.high,
                "num_scenes": difference.num_scenes,
                "seeds": " ".join(map(str, comparison.seeds)),
                "partial": comparison.partial,
                "software": comparison.software,
                "verdict": comparison.verdict,
            }
        )
    # The csv module already ends rows with "\r\n"; newline="" stops Windows adding another "\r".
    write_atomically(path, lambda temporary_path: temporary_path.write_text(file.getvalue(), encoding="utf-8", newline=""))


def _write_markdown(
    suite_name: str,
    results: list[GroupResult],
    seeds: list[int],
    comparisons: list[_Comparison],
    figures: list[tuple[str, Path]],
    failures: Mapping[str, str],
    differences: dict[str, dict],
    path: Path,
) -> None:
    multiple_seeds = len(seeds) > 1
    lines = [
        f"# Suite report: {suite_name}",
        "",
        f"Updated {datetime.now():%Y-%m-%d %H:%M}. Bold marks the best value of each quality metric.",
    ]
    if multiple_seeds:
        lines += [
            "",
            f"Every run was trained {len(seeds)} times, with seeds {', '.join(map(str, seeds))}. Quality is shown "
            "as mean ± standard deviation over the seeds; times and speeds are means. A difference between "
            "runs smaller than their spread over seeds may be noise.",
        ]
    lines += [
        "",
        "## Status and cost",
        "",
        "| Run | Status | Training steps | Training time (h:mm) | Parameters | Rendering FPS |",
        "|---|---|--:|--:|--:|--:|",
    ]
    for group in results:
        lines.append(
            f"| {group.name} | {group.status} | {group.training_steps_text} "
            f"| {_format_duration(group.training_hours)} "
            f"| {_format_count(group.parameters)} "
            f"| {_format(group.frames_per_second, '{:.2f}')} |"
        )

    left_out = [(run.key, reason) for group in results for run in group.replicates for reason in run.left_out]
    if left_out:
        lines += [
            "",
            "Some saved results no longer match their run's current checkpoint or settings, or are damaged. "
            "They are left out of every table and figure; running the suite again redoes them:",
            "",
            *[f"- **{key}**, {reason}" for key, reason in left_out],
        ]

    lines += ["", *_differences_section(results, differences, multiple_seeds)]

    metric_names = _metric_names(results)
    for count in _view_counts(results):
        lines += ["", f"## Quality with {_views(count)}", ""]
        lines.append("| Run | " + " | ".join(_metric_header(metric) for metric in metric_names) + " |")
        lines.append("|---|" + "--:|" * len(metric_names))
        best = {metric: _best_value(results, count, metric) for metric in metric_names}
        for group in results:
            if not any(count in run.metrics for run in group.with_results):
                continue
            cells = []
            for metric in metric_names:
                summary = group.summary(count, metric)
                interim_summary = group.summary(count, metric, interim=True)
                if summary is None and interim_summary is not None:
                    # Shown for information only; marked and never "best".
                    cells.append(_format_summary(interim_summary, metric, len(seeds)) + " *")
                    continue
                text = _format_summary(summary, metric, len(seeds))
                is_best = summary is not None and summary.mean == best[metric]
                cells.append(f"**{text}**" if is_best else text)
            lines.append(f"| {group.name} | " + " | ".join(cells) + " |")

    interim = [(run.key, run.interim_reason) for group in results for run in group.interim]
    if interim:
        lines += [
            "",
            "\\* Interim: evaluated before the requested training finished. Shown for information only; left out "
            "of best values, figures and paired comparisons:",
            "",
            *[f"- **{key}**: {reason}" for key, reason in interim],
        ]

    lines += _comparison_section(results, comparisons, metric_names, multiple_seeds)

    if figures:
        lines += ["", "## Figures", ""]
        for title, figure_path in figures:
            # Markdown links use forward slashes, also on Windows.
            link = figure_path.relative_to(path.parent).as_posix()
            lines += [f"### {title}", "", f"![{title}]({link})", ""]

    if failures:
        lines += ["", "## Failed runs", "", "Full error messages are in the suite log.", ""]
        lines += [f"- **{key}**: `{message}`" for key, message in failures.items()]

    write_atomically(path, lambda temporary_path: temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8"))


def _differences_section(results: list[GroupResult], differences: dict[str, dict], multiple_seeds: bool) -> list[str]:
    """What was varied between the runs, so the reader knows everything else was held fixed.

    Whether the runs were trained on identical data is judged from what their
    checkpoints recorded, not only from the current config: a dataset can be
    replaced in place between two runs.
    """
    lines = ["## What differs between runs", ""]
    recorded = [run.training_data for group in results for run in group.replicates]
    all_recorded = all(fingerprint is not None for fingerprint in recorded)
    recorded_data_differs = len({_canonical_json(fp) for fp in recorded if fp is not None}) > 1
    weights_by_source = _pretrained_weights_by_source(results)
    weights_differ = [source for source, hashes in weights_by_source.items() if len(hashes) > 1]
    evaluated = [run for group in results for run in group.replicates if run.metrics]
    # Also for a single evaluation: one that cannot be compared is flagged while the others are pending.
    software = _compare_software(evaluated) if evaluated else SOFTWARE_SAME
    software_differs = software != SOFTWARE_SAME

    if not differences and not recorded_data_differs and not weights_differ and not software_differs:
        lines.append("Nothing: all runs use identical settings.")
    else:
        lines.append(
            "Every setting not listed here is identical for all runs. The measurement settings "
            "(evaluation, validation, profiling, the validation and test data, and renderer.ray_chunk_size) "
            "are always identical; the suite does not start otherwise."
        )
    if recorded_data_differs:
        lines.append(
            "**Warning: the runs' checkpoints record different training data** (see the last row of the "
            "table). Their results are not a comparison on equal data."
        )
    elif trained_on_identical_data(differences):
        if all_recorded:
            lines.append(
                "For each seed, all runs were trained on identical data, step for step, as their checkpoints confirm."
                if multiple_seeds
                else "All runs were trained on identical data, step for step, as their checkpoints confirm."
            )
        else:
            lines.append(
                "All runs are configured to train on identical data, step for step; this is confirmed from "
                "their checkpoints once every run has been trained."
            )
    for source in weights_differ:
        lines.append(
            f"**Warning: runs that start from `{source}` started from different weights** (see the row "
            "\"pretrained weights\" of the table): the file was replaced between runs, or exists in several versions."
        )
    if software == SOFTWARE_DIFFERENT:
        lines.append(
            "**Warning: the runs were evaluated with different software** (see the row \"evaluation software "
            "that differs\" of the table). Small differences in quality may come from it rather than from the "
            "models. The report compares every installed package, so a listed package may not matter for a "
            "score. For an exact comparison, evaluate every run again with one installation (`scene-recon evaluate`)."
        )
    elif software == SOFTWARE_UNKNOWN:
        found = {problem for run in evaluated for problem in run.software_problems}
        problems = [problem for problem in SOFTWARE_PROBLEMS if problem in found]
        lines.append(
            "**Warning: the software of some evaluations cannot be compared** (see the row \"evaluation software "
            "that differs\" of the table), so the report cannot tell whether all runs were scored with the same "
            "software. "
            + " ".join(f"*{problem}*: {SOFTWARE_PROBLEMS[problem]}." for problem in problems)
        )
    if not differences and not recorded_data_differs and not weights_by_source and not software_differs:
        return lines

    lines += [
        "",
        "| Setting | " + " | ".join(group.name for group in results) + " |",
        "|---|" + "---|" * len(results),
    ]
    for setting, values in differences.items():
        cells = [_table_cell(values.get(group.name, MISSING)) for group in results]
        lines.append(f"| `{setting}` | " + " | ".join(cells) + " |")
    if recorded_data_differs:
        cells = [_describe_training_data(group) for group in results]
        lines.append("| training data recorded in the checkpoint | " + " | ".join(cells) + " |")
    if weights_by_source:
        cells = [_by_seed(group, _describe_starting_weights) for group in results]
        lines.append("| pretrained weights recorded in the checkpoint | " + " | ".join(cells) + " |")
    if software_differs:
        names = _differing_software(evaluated)
        cells = [_by_seed(group, lambda run: _describe_evaluation_software(run, names)) for group in results]
        lines.append("| evaluation software that differs | " + " | ".join(cells) + " |")
    return lines


def _by_seed(group: GroupResult, describe) -> str:
    """``describe(run)`` for a run's seeds: one entry if they agree, otherwise one per seed, labelled with it."""
    by_seed = {run.seed: describe(run) for run in group.replicates}
    if len(set(by_seed.values())) <= 1:
        return next(iter(by_seed.values()), MISSING)
    return "; ".join(f"seed {seed}: {text}" for seed, text in by_seed.items())


def _pretrained_weights_by_source(results: list[GroupResult]) -> dict[str, set[str]]:
    """Every pretrained-weights source the runs' checkpoints record, with the hashes recorded for it."""
    hashes: dict[str, set[str]] = {}
    for group in results:
        for run in group.replicates:
            for weights in (run.starting_weights or {}).values():
                hashes.setdefault(weights["source"], set()).add(weights["sha256"])
    return hashes


def _describe_starting_weights(run: RunResult) -> str:
    """A short, readable identifier of the pretrained weights a run's checkpoint started from."""
    if run.starting_weights is None:
        return MISSING
    if not run.starting_weights:
        return "none (random start)"
    return ", ".join(
        f"`{part}: {Path(weights['source']).name} {weights['sha256'][:12]}`"
        for part, weights in run.starting_weights.items()
    )


def _differing_software(runs: list[RunResult]) -> list[str]:
    """The labels whose versions are not the same in every readable evaluation record of these runs (absent counts).

    Runtime fields come first, then packages in alphabetical order.
    """
    records = [_software_for_display(run.evaluation_software) for run in runs if run.evaluation_software is not None]
    runtime = list(_RUNTIME_LABELS.values())
    packages = sorted({name for record in records for name in record} - set(runtime))
    return [name for name in runtime + packages if len({record.get(name) for record in records}) > 1]


def _describe_evaluation_software(run: RunResult, names: list[str]) -> str:
    """The versions of ``names`` a run's evaluation was made with, or why its record cannot be compared."""
    if not run.metrics:
        return MISSING
    if run.evaluation_software is None:
        return ", ".join(run.software_problems)
    software = _software_for_display(run.evaluation_software)
    return ", ".join(f"{name} {software.get(name, 'absent')}" for name in names) or "–"


def _short_hash(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()[:12]


def _describe_training_data(group: GroupResult) -> str:
    """A short, readable identifier of the training data in a run's checkpoints."""
    descriptions = []
    for run in group.replicates:
        fingerprint = run.training_data
        if fingerprint is None:
            text = MISSING
        elif "content_sha256" in fingerprint:
            text = f"`{fingerprint.get('dataset', '')} {fingerprint['content_sha256'][:12]}`"
        else:
            text = _table_cell(fingerprint)
        if text not in descriptions:
            descriptions.append(text)
    return ", ".join(descriptions)


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True)


def _comparison_section(
    results: list[GroupResult], comparisons: list[_Comparison], metric_names: list[str], multiple_seeds: bool
) -> list[str]:
    if len(results) < 2:
        return []
    reference = results[0].name
    lines = ["", f"## Paired comparison with {reference}", ""]
    not_compared = [
        f"- **{group.name}**: " + "; ".join(run.interim_reason for run in group.interim)
        for group in results
        if group.interim and not group.eligible
    ]
    if not comparisons:
        lines.append(
            f"Not available yet: the reference run {reference} (the first run of the suite file) and at least "
            "one other run need current evaluations of the same seeds, made after their requested training finished."
        )
        return lines + (["", "Waiting for:", "", *not_compared] if not_compared else [])

    lines.append(
        f"Every model is evaluated on the same test scenes, so runs are compared scene by scene: for each "
        f"scene, a run's score minus {reference}'s, averaged over the scenes, with its 95% confidence interval. "
        "A run is called better or worse only when the interval excludes zero."
    )
    if multiple_seeds:
        lines.append(
            "Only seeds that both runs have results for are compared, and each run's per-scene scores are "
            "averaged over those seeds. *Partial* marks a comparison that does not yet cover every seed."
        )

    for count in sorted({comparison.count for comparison in comparisons}):
        runs = list(dict.fromkeys(c.run for c in comparisons if c.count == count))
        seeds_header = " Seeds compared |" if multiple_seeds else ""
        lines += [
            "",
            f"### {_views(count).capitalize()}",
            "",
            "| Run |" + seeds_header + " " + " | ".join(_metric_header(metric) for metric in metric_names) + " |",
            "|---|" + ("---|" if multiple_seeds else "") + "---|" * len(metric_names),
        ]
        for run in runs:
            matches = {c.metric: c for c in comparisons if c.run == run and c.count == count}
            cells = [_format_difference(matches[metric]) if metric in matches else "–" for metric in metric_names]
            seeds_cell = ""
            if multiple_seeds:
                first = next(iter(matches.values()))
                seeds_cell = " " + ", ".join(map(str, first.seeds)) + (" (partial)" if first.partial else "") + " |"
            lines.append(f"| {run} |" + seeds_cell + " " + " | ".join(cells) + " |")
    if any(comparison.software != SOFTWARE_SAME for comparison in comparisons):
        lines += [
            "",
            "A verdict marked *scored with different software* compares evaluations made with different software, "
            "and one marked *unknown* evaluations whose software cannot be compared (see \"What differs between "
            "runs\"). The difference may then come from the software; evaluate both runs again with one "
            "installation before drawing a conclusion.",
        ]
    if not_compared:
        lines += ["", "Not compared, because their requested training has not finished:", "", *not_compared]
    return lines


def _format_difference(comparison: _Comparison) -> str:
    difference = comparison.difference
    pattern = "{:+.2f}" if comparison.metric == "psnr" else "{:+.3f}"
    interval = f"[{pattern.format(difference.low)}, {pattern.format(difference.high)}]"
    return f"{pattern.format(difference.mean)} {interval} {comparison.verdict}"


def _format_summary(summary: Summary | None, metric: str, num_seeds: int) -> str:
    if summary is None:
        return "–"
    pattern = "{:.2f}" if metric == "psnr" else "{:.3f}"
    if num_seeds == 1:
        return pattern.format(summary.mean)
    text = f"{pattern.format(summary.mean)} ± {pattern.format(summary.std)}"
    return text if summary.count == num_seeds else f"{text} (n={summary.count})"


def _views(count: int) -> str:
    return f"{count} input view{'s' if count > 1 else ''}"


def _table_cell(value) -> str:
    if value == MISSING:
        return MISSING
    return f"`{format_value(value)}`".replace("|", "\\|")


def _metric_header(metric: str) -> str:
    label, higher_is_better = _metric_presentation(metric)
    if higher_is_better is None:
        return label
    return f"{label} {'↑' if higher_is_better else '↓'}"


def _metric_presentation(metric: str) -> tuple[str, bool | None]:
    """(display name, whether higher is better) as declared by the metric's class."""
    try:
        metric_class = METRICS.get(metric)
    except ConfigError:  # a metric that is no longer registered
        return metric, None
    return getattr(metric_class, "display_name", metric), getattr(metric_class, "higher_is_better", None)


def _best_value(results: list[GroupResult], count: int, metric: str) -> float | None:
    """The best mean among the runs, or ``None`` if there is nothing to compare."""
    _, higher_is_better = _metric_presentation(metric)
    means = [summary.mean for group in results if (summary := group.summary(count, metric)) is not None]
    if higher_is_better is None or len(means) < 2:
        return None
    return max(means) if higher_is_better else min(means)


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _format(value, pattern: str) -> str:
    return "–" if value is None else pattern.format(value)


def _format_duration(hours: float | None) -> str:
    """Hours as ``h:mm``, e.g. 1.5 -> ``1:30``."""
    if hours is None:
        return "–"
    total_minutes = round(hours * 60)
    return f"{total_minutes // 60}:{total_minutes % 60:02d}"


def _format_count(count: int | None) -> str:
    """Parameter counts: exact below a million, in millions above."""
    if count is None:
        return "–"
    return f"{count:,}" if count < 1_000_000 else f"{count / 1e6:.2f} M"


# ---------------------------------------------------------------- figures


def _write_figures(results: list[GroupResult], plots_dir: Path, multiple_seeds: bool) -> list[tuple[str, Path]]:
    """Draw every figure there is data for. Returns (title, path) of each figure written."""
    styles = run_styles([group.name for group in results])
    metric_names = _metric_names(results)
    headline = _HEADLINE_METRIC if _HEADLINE_METRIC in metric_names else (metric_names or [None])[0]
    evaluated = [group for group in results if group.eligible]
    figures = []

    if evaluated:
        means: dict[str, dict[int, dict[str, float]]] = {}
        spreads: dict[str, dict[int, dict[str, float]]] = {}
        for group in evaluated:
            for count in _view_counts([group]):
                for metric in metric_names:
                    summary = group.summary(count, metric)
                    if summary is not None:
                        means.setdefault(group.name, {}).setdefault(count, {})[metric] = summary.mean
                        spreads.setdefault(group.name, {}).setdefault(count, {})[metric] = summary.std
        path = plots_dir / "metrics_vs_source_views.png"
        plot_metrics_vs_source_views(
            means,
            {metric: _axis_label(metric) for metric in metric_names},
            styles,
            path,
            spread=spreads if multiple_seeds else None,
        )
        title = "Quality against the number of input views"
        figures.append((f"{title} (error bars: standard deviation over seeds)" if multiple_seeds else title, path))

    profiled = [group for group in evaluated if group.frames_per_second is not None]
    if profiled and headline is not None:
        path = plots_dir / "quality_vs_speed.png"
        points = {}
        for group in profiled:
            per_count = {}
            for count in _view_counts([group]):
                summary = group.summary(count, headline)
                if summary is not None:
                    per_count[count] = summary.mean
            points[group.name] = (group.frames_per_second, per_count)
        plot_quality_vs_speed(points, _axis_label(headline), styles, path)
        figures.append(("Quality against rendering speed", path))

    curves = {}
    for group in results:
        per_seed = [_read_training_curves(run.files, headline) for run in group.replicates]
        group_curves = _mean_curves([run_curves for run_curves in per_seed if run_curves])
        if group_curves:
            curves[group.name] = group_curves
    if curves:
        path = plots_dir / "training_curves.png"
        validation_label = f"Validation {_metric_presentation(headline)[0]}" if headline else "Validation"
        plot_training_curves(curves, validation_label, styles, path)
        figures.append(("Training curves (mean over seeds)" if multiple_seeds else "Training curves", path))

    for count in _view_counts(results):
        path = plots_dir / f"renders_{count}_source_views.png"
        if _plot_render_comparison(evaluated, count, path):
            title = f"Renders with {_views(count)}"
            figures.append((f"{title} (first seed of each run)" if multiple_seeds else title, path))

    return figures


def _axis_label(metric: str) -> str:
    label, higher_is_better = _metric_presentation(metric)
    if higher_is_better is None:
        return label
    return f"{label}, {'higher' if higher_is_better else 'lower'} is better"


def _read_training_curves(files: RunFiles, validation_metric: str | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read the logged training curves of one run from its TensorBoard files."""
    if not files.tensorboard.is_dir():
        return {}

    # Imported here because loading TensorBoard's reader takes a moment and
    # is only needed when there are curves to draw.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logging.getLogger("tensorboard").setLevel(logging.WARNING)
    accumulator = EventAccumulator(str(files.tensorboard), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])

    tags = {"loss": "loss/total", "hours": "time/training_hours"}
    if validation_metric is not None:
        tags["validation"] = f"validation/{validation_metric}"

    curves = {}
    for name, tag in tags.items():
        if tag not in available:
            continue
        # A resumed run logs some steps twice; the later value wins.
        by_step = {event.step: event.value for event in accumulator.Scalars(tag)}
        steps = np.array(sorted(by_step))
        curves[name] = (steps, np.array([by_step[step] for step in steps]))
    return curves


def _mean_curves(per_seed: list[dict[str, tuple[np.ndarray, np.ndarray]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Average each curve over seeds, at the steps every seed has logged."""
    averaged = {}
    for name in {name for curves in per_seed for name in curves}:
        available = [curves[name] for curves in per_seed if name in curves]
        steps = reduce(np.intersect1d, [curve_steps for curve_steps, _ in available])
        if len(steps) == 0:
            continue
        values = [curve_values[np.searchsorted(curve_steps, steps)] for curve_steps, curve_values in available]
        averaged[name] = (steps, np.mean(values, axis=0))
    return averaged


def _plot_render_comparison(results: list[GroupResult], count: int, path: Path) -> bool:
    """Show the preview renders of every run (its first evaluated seed) for one protocol.

    Returns whether a figure was written.
    """
    shown = []
    for group in results:
        run = next((run for run in group.eligible if run.preview_scene_ids.get(count)), None)
        if run is not None:
            shown.append((group.name, run))
    if not shown:
        return False

    # Scenes that every run has a preview of, in the order of the first run.
    scene_ids = [
        scene_id
        for scene_id in shown[0][1].preview_scene_ids[count]
        if all(scene_id in run.preview_scene_ids[count] for _, run in shown)
    ]
    if not scene_ids:
        return False

    column_titles = [f"Input {index + 1}" for index in range(count)] + ["Ground truth"]
    column_titles += [name for name, _ in shown]
    rows = []
    for scene_id in scene_ids:
        previews = [load_preview_images(run.files.previews(count) / scene_id) for _, run in shown]
        rows.append(previews[0]["sources"] + [previews[0]["truth"]] + [preview["prediction"] for preview in previews])

    plot_image_grid(column_titles, rows, scene_ids, path)
    return True
