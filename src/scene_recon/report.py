"""The suite report: tables and figures comparing every run of a suite.

Written into the suite directory and rewritten after every run, so partial
results can be inspected while later runs are still training::

    summary.md                                status, cost and quality tables, with the figures
    summary.csv                               the same numbers, one row per run, for your own analysis
    plots/metrics_vs_source_views.png         each metric against the number of input views
    plots/quality_vs_speed.png                quality against rendering speed
    plots/training_curves.png                 training loss and validation quality over training
    plots/renders_<n>_source_views.png        the same views rendered by every run, next to the truth

The report only reads the files each run wrote (see ``scene_recon.run_files``),
so it can be regenerated at any time without touching the models.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import scene_recon.metrics  # noqa: F401  (registers the metrics, whose display settings are read below)
from scene_recon.checkpoints import read_progress
from scene_recon.config import Config
from scene_recon.evaluation import load_preview_images
from scene_recon.fairness import MISSING, differing_settings, format_value, trained_on_identical_data
from scene_recon.plots import (
    plot_image_grid,
    plot_metrics_vs_source_views,
    plot_quality_vs_speed,
    plot_training_curves,
    run_styles,
)
from scene_recon.registry import METRICS, ConfigError
from scene_recon.run_files import RunFiles

logger = logging.getLogger(__name__)

# The metric used for the quality-versus-speed figure and the validation
# curves, if the runs report it; otherwise the first metric is used.
_HEADLINE_METRIC = "psnr"


@dataclass
class RunResult:
    """Everything the report shows about one run, read from the run's output files."""

    name: str
    status: str
    files: RunFiles
    training_step: int | None = None
    training_hours: float | None = None
    parameters: int | None = None
    frames_per_second: float | None = None
    metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    """Number of source views -> metric name -> mean over the test scenes."""

    preview_scene_ids: dict[int, list[str]] = field(default_factory=dict)
    """Number of source views -> scenes with a saved preview render."""


def write_report(suite_name: str, directory: Path, runs: dict[str, Config], failures: dict[str, str]) -> None:
    """Write the tables and figures of a suite.

    Args:
        suite_name: Shown as the report's title.
        directory: The suite directory.
        runs: Every run of the suite, in suite order.
        failures: Runs that failed, mapped to their error message.
    """
    results = [_collect(name, config, failed=name in failures) for name, config in runs.items()]
    plots_dir = directory / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(results, directory / "summary.csv")
    figures = _write_figures(results, plots_dir)
    _write_markdown(suite_name, results, figures, failures, differing_settings(runs), directory / "summary.md")
    logger.info("Suite report updated: %s", directory / "summary.md")


def _collect(name: str, config: Config, failed: bool) -> RunResult:
    files = RunFiles(config.run_dir)
    result = RunResult(name=name, status="not started", files=files)

    progress = read_progress(files.checkpoint)
    if progress is not None:
        result.training_step = progress.step
        result.training_hours = progress.training_seconds / 3600
        result.status = "trained" if progress.step >= config.training.num_steps else "training"

    if files.evaluation.is_file():
        evaluation = json.loads(files.evaluation.read_text())
        for protocol in evaluation["protocols"]:
            count = len(protocol["source_views"])
            result.metrics[count] = protocol["mean"]
            result.preview_scene_ids[count] = protocol.get("preview_scene_ids", [])
        result.status = "evaluated"

    if files.profile.is_file():
        profile = json.loads(files.profile.read_text())
        result.parameters = profile["parameters"]["total"]
        result.frames_per_second = profile["rendering"]["frames_per_second"]

    if failed:
        result.status = "failed"
    return result


# ---------------------------------------------------------------- tables


def _metric_names(results: list[RunResult]) -> list[str]:
    """Every metric reported by any run, in the order the runs list them."""
    names: list[str] = []
    for result in results:
        for per_metric in result.metrics.values():
            names.extend(name for name in per_metric if name not in names)
    return names


def _view_counts(results: list[RunResult]) -> list[int]:
    return sorted({count for result in results for count in result.metrics})


def _write_csv(results: list[RunResult], path: Path) -> None:
    """One row per run. Quality columns are named ``<metric>_<n>v``, e.g. ``psnr_1v``."""
    metric_names, view_counts = _metric_names(results), _view_counts(results)
    quality_columns = [f"{metric}_{count}v" for count in view_counts for metric in metric_names]
    columns = ["run", "status", "training_steps", "training_hours", "parameters", "frames_per_second"]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns + quality_columns)
        writer.writeheader()
        for result in results:
            row = {
                "run": result.name,
                "status": result.status,
                "training_steps": result.training_step,
                "training_hours": result.training_hours,
                "parameters": result.parameters,
                "frames_per_second": result.frames_per_second,
            }
            for count, per_metric in result.metrics.items():
                for metric, value in per_metric.items():
                    row[f"{metric}_{count}v"] = value
            writer.writerow(row)


def _write_markdown(
    suite_name: str,
    results: list[RunResult],
    figures: list[tuple[str, Path]],
    failures: dict[str, str],
    differences: dict[str, dict],
    path: Path,
) -> None:
    lines = [
        f"# Suite report: {suite_name}",
        "",
        f"Updated {datetime.now():%Y-%m-%d %H:%M}. Bold marks the best value of each quality metric.",
        "",
        "## Status and cost",
        "",
        "| Run | Status | Training steps | Training time (h:mm) | Parameters | Rendering FPS |",
        "|---|---|--:|--:|--:|--:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.status} | {_format(result.training_step, '{:,}')} "
            f"| {_format_duration(result.training_hours)} "
            f"| {_format_count(result.parameters)} "
            f"| {_format(result.frames_per_second, '{:.2f}')} |"
        )

    lines += ["", *_differences_section(results, differences)]

    metric_names = _metric_names(results)
    for count in _view_counts(results):
        lines += ["", f"## Quality with {count} input view{'s' if count > 1 else ''}", ""]
        lines.append("| Run | " + " | ".join(_metric_header(metric) for metric in metric_names) + " |")
        lines.append("|---|" + "--:|" * len(metric_names))
        best = {metric: _best_value(results, count, metric) for metric in metric_names}
        for result in results:
            if count not in result.metrics:
                continue
            cells = []
            for metric in metric_names:
                value = result.metrics[count].get(metric)
                text = _format(value, "{:.3f}" if metric != "psnr" else "{:.2f}")
                cells.append(f"**{text}**" if value is not None and value == best[metric] else text)
            lines.append(f"| {result.name} | " + " | ".join(cells) + " |")

    if figures:
        lines += ["", "## Figures", ""]
        for title, figure_path in figures:
            lines += [f"### {title}", "", f"![{title}]({figure_path.relative_to(path.parent)})", ""]

    if failures:
        lines += ["", "## Failed runs", "", "Full error messages are in the suite log.", ""]
        lines += [f"- **{name}**: `{message}`" for name, message in failures.items()]

    path.write_text("\n".join(lines) + "\n")


def _differences_section(results: list[RunResult], differences: dict[str, dict]) -> list[str]:
    """What was varied between the runs, so the reader knows everything else was held fixed."""
    lines = ["## What differs between runs", ""]
    if not differences:
        return lines + ["Nothing: all runs use identical settings."]

    lines.append(
        "Every setting not listed here is identical for all runs. The measurement settings "
        "(evaluation, validation, profiling, and the validation and test data) are always identical; "
        "the suite does not start otherwise."
    )
    if trained_on_identical_data(differences):
        lines.append("All runs were trained on identical data, step for step.")
    lines += [
        "",
        "| Setting | " + " | ".join(result.name for result in results) + " |",
        "|---|" + "---|" * len(results),
    ]
    for setting, values in differences.items():
        cells = [_table_cell(values.get(result.name, MISSING)) for result in results]
        lines.append(f"| `{setting}` | " + " | ".join(cells) + " |")
    return lines


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


def _best_value(results: list[RunResult], count: int, metric: str) -> float | None:
    _, higher_is_better = _metric_presentation(metric)
    values = [result.metrics[count][metric] for result in results if metric in result.metrics.get(count, {})]
    if higher_is_better is None or len(values) < 2:
        return None
    return max(values) if higher_is_better else min(values)


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


def _write_figures(results: list[RunResult], plots_dir: Path) -> list[tuple[str, Path]]:
    """Draw every figure there is data for. Returns (title, path) of each figure written."""
    styles = run_styles([result.name for result in results])
    metric_names = _metric_names(results)
    headline = _HEADLINE_METRIC if _HEADLINE_METRIC in metric_names else (metric_names or [None])[0]
    figures = []

    evaluated = [result for result in results if result.metrics]
    if evaluated:
        path = plots_dir / "metrics_vs_source_views.png"
        plot_metrics_vs_source_views(
            {result.name: result.metrics for result in evaluated},
            {metric: _axis_label(metric) for metric in metric_names},
            styles,
            path,
        )
        figures.append(("Quality against the number of input views", path))

    profiled = [result for result in evaluated if result.frames_per_second is not None]
    if profiled and headline is not None:
        path = plots_dir / "quality_vs_speed.png"
        plot_quality_vs_speed(
            {
                result.name: (
                    result.frames_per_second,
                    {count: values[headline] for count, values in result.metrics.items() if headline in values},
                )
                for result in profiled
            },
            _axis_label(headline),
            styles,
            path,
        )
        figures.append(("Quality against rendering speed", path))

    curves = {result.name: _read_training_curves(result.files, headline) for result in results}
    curves = {name: run_curves for name, run_curves in curves.items() if run_curves}
    if curves:
        path = plots_dir / "training_curves.png"
        validation_label = f"Validation {_metric_presentation(headline)[0]}" if headline else "Validation"
        plot_training_curves(curves, validation_label, styles, path)
        figures.append(("Training curves", path))

    for count in _view_counts(results):
        path = plots_dir / f"renders_{count}_source_views.png"
        if _plot_render_comparison(evaluated, count, path):
            figures.append((f"Renders with {count} input view{'s' if count > 1 else ''}", path))

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


def _plot_render_comparison(results: list[RunResult], count: int, path: Path) -> bool:
    """Show the preview renders of every run for one protocol. Returns whether a figure was written."""
    with_previews = [result for result in results if result.preview_scene_ids.get(count)]
    if not with_previews:
        return False

    # Scenes that every run has a preview of, in the order of the first run.
    scene_ids = [
        scene_id
        for scene_id in with_previews[0].preview_scene_ids[count]
        if all(scene_id in result.preview_scene_ids[count] for result in with_previews)
    ]
    if not scene_ids:
        return False

    column_titles = [f"Input {index + 1}" for index in range(count)] + ["Ground truth"]
    column_titles += [result.name for result in with_previews]
    rows = []
    for scene_id in scene_ids:
        previews = [load_preview_images(result.files.previews(count) / scene_id) for result in with_previews]
        rows.append(previews[0]["sources"] + [previews[0]["truth"]] + [preview["prediction"] for preview in previews])

    plot_image_grid(column_titles, rows, scene_ids, path)
    return True
