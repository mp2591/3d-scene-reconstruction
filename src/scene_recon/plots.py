"""Figures for the suite report.

Each function draws one figure from plain dictionaries keyed by run name and
saves it as a PNG. None of them reads files, so they are easy to reuse in a
notebook.

Every run keeps one style across all figures: a colour from a fixed
categorical order, and its own marker shape, so runs can also be told apart
without relying on colour (colour-blind readers, greyscale printing). Colours
and ink follow the reference palette of the data-visualisation guidelines the
report was designed with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Categorical colours, assigned to runs in this order (never reordered, so a
# run keeps its colour when other runs are added or fail).
_SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"

_STYLE = {
    "figure.facecolor": _SURFACE,
    "savefig.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "axes.edgecolor": _AXIS,
    "axes.labelcolor": _INK_SECONDARY,
    "axes.titlecolor": _INK,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": _GRIDLINE,
    "grid.linewidth": 0.8,
    "xtick.color": _AXIS,
    "ytick.color": _AXIS,
    "xtick.labelcolor": _INK_MUTED,
    "ytick.labelcolor": _INK_MUTED,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "legend.frameon": False,
    "legend.labelcolor": _INK_SECONDARY,
    "font.size": 10,
    "text.color": _INK,
}
_DPI = 150


@dataclass(frozen=True)
class RunStyle:
    color: str
    marker: str
    linestyle: str


def run_styles(run_names: Sequence[str]) -> dict[str, RunStyle]:
    """Give each run its colour and marker, by its position in the suite.

    Past eight runs the colours repeat; those runs are drawn with dashed
    lines, so a repeated colour is never the only distinction.
    """
    return {
        name: RunStyle(
            color=_SERIES_COLORS[index % len(_SERIES_COLORS)],
            marker=_MARKERS[index % len(_MARKERS)],
            linestyle="-" if index < len(_SERIES_COLORS) else "--",
        )
        for index, name in enumerate(run_names)
    }


def plot_metrics_vs_source_views(
    metrics: dict[str, dict[int, dict[str, float]]],
    metric_labels: dict[str, str],
    styles: dict[str, RunStyle],
    path: Path,
) -> None:
    """One panel per metric: its value against the number of input views, one line per run.

    Args:
        metrics: Run name -> number of source views -> metric name -> value.
        metric_labels: Metric name -> panel title, in panel order.
        styles: From ``run_styles``.
        path: Where to save the PNG.
    """
    view_counts = sorted({count for per_run in metrics.values() for count in per_run})
    with matplotlib.rc_context(_STYLE):
        figure = Figure(figsize=(4.0 * len(metric_labels), 3.6), layout="constrained", dpi=_DPI)
        axes = figure.subplots(1, len(metric_labels), squeeze=False)[0]
        for ax, (metric, label) in zip(axes, metric_labels.items()):
            for run_name, per_count in metrics.items():
                counts = [count for count in view_counts if metric in per_count.get(count, {})]
                values = [per_count[count][metric] for count in counts]
                style = styles[run_name]
                ax.plot(
                    counts, values, color=style.color, marker=style.marker, linestyle=style.linestyle, label=run_name
                )
            ax.set_title(label)
            ax.set_xticks(view_counts)
            ax.set_xlabel("Input views")
        _add_legend(figure, styles, list(metrics))
        figure.savefig(path)


def plot_quality_vs_speed(
    points: dict[str, tuple[float, dict[int, float]]],
    quality_label: str,
    styles: dict[str, RunStyle],
    path: Path,
) -> None:
    """One panel per number of input views: quality against rendering speed, one point per run.

    Args:
        points: Run name -> (frames per second, number of source views -> quality value).
        quality_label: Axis label for the quality value, e.g. "PSNR (dB), higher is better".
        styles: From ``run_styles``.
        path: Where to save the PNG.
    """
    view_counts = sorted({count for _, per_count in points.values() for count in per_count})
    speeds = [fps for fps, _ in points.values()]
    use_log_scale = max(speeds) / max(min(speeds), 1e-9) > 10

    with matplotlib.rc_context(_STYLE):
        figure = Figure(figsize=(4.0 * len(view_counts), 3.8), layout="constrained", dpi=_DPI)
        axes = figure.subplots(1, len(view_counts), squeeze=False, sharey=True)[0]
        for ax, count in zip(axes, view_counts):
            for run_name, (fps, per_count) in points.items():
                if count not in per_count:
                    continue
                style = styles[run_name]
                ax.plot(fps, per_count[count], color=style.color, marker=style.marker, linestyle="none", label=run_name)
                ax.annotate(
                    run_name,
                    (fps, per_count[count]),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=_INK_SECONDARY,
                )
            if use_log_scale:
                ax.set_xscale("log")
            ax.grid(True, axis="both")
            ax.set_title(f"{count} input view{'s' if count > 1 else ''}")
            ax.set_xlabel("Rendering speed (frames per second)")
        axes[0].set_ylabel(quality_label)
        _add_legend(figure, styles, list(points), with_lines=False)
        figure.savefig(path)


def plot_training_curves(
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    validation_label: str,
    styles: dict[str, RunStyle],
    path: Path,
) -> None:
    """Training loss and validation quality over the course of training.

    Panels: training loss by step, validation quality by step, and validation
    quality by training hours (the convergence-efficiency comparison of the
    proposal). Panels without data for any run are left out.

    Args:
        curves: Run name -> curve name -> (steps, values). Curve names are
            ``"loss"``, ``"validation"`` and ``"hours"`` (training hours at each logged step).
        validation_label: What the validation curve measures, e.g. "Validation PSNR (dB)".
        styles: From ``run_styles``.
        path: Where to save the PNG.
    """
    panels = []
    if any("loss" in run for run in curves.values()):
        panels.append("loss")
    if any("validation" in run for run in curves.values()):
        panels.append("validation_by_step")
        if any("hours" in run for run in curves.values()):
            panels.append("validation_by_hours")

    # Short runs are shown in minutes, so the time axis has readable numbers.
    longest_hours = max((run["hours"][1].max() for run in curves.values() if "hours" in run), default=0.0)
    time_unit, time_scale = ("hours", 1.0) if longest_hours >= 1.0 else ("minutes", 60.0)

    with matplotlib.rc_context(_STYLE):
        figure = Figure(figsize=(4.2 * len(panels), 3.6), layout="constrained", dpi=_DPI)
        axes = figure.subplots(1, len(panels), squeeze=False)[0]
        for ax, panel in zip(axes, panels):
            for run_name, run_curves in curves.items():
                style = styles[run_name]
                if panel == "loss":
                    if "loss" not in run_curves:
                        continue
                    steps, values = _moving_average(*run_curves["loss"])
                else:
                    if "validation" not in run_curves:
                        continue
                    steps, values = run_curves["validation"]
                    if panel == "validation_by_hours":
                        if "hours" not in run_curves:
                            continue
                        steps = np.interp(steps, *run_curves["hours"]) * time_scale
                # About eight markers per line, so lines differ by shape as well as colour.
                ax.plot(
                    steps,
                    values,
                    color=style.color,
                    linestyle=style.linestyle,
                    marker=style.marker,
                    markevery=max(1, len(steps) // 8),
                    label=run_name,
                )
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

            if panel == "loss":
                ax.set_yscale("log")
                ax.set_title("Training loss (moving average)")
                ax.set_xlabel("Step")
            elif panel == "validation_by_step":
                ax.set_title(validation_label)
                ax.set_xlabel("Step")
            else:
                ax.set_title(f"{validation_label} by training time")
                ax.set_xlabel(f"Training {time_unit}")
        _add_legend(figure, styles, list(curves))
        figure.savefig(path)


def plot_image_grid(
    column_titles: list[str],
    rows: list[list[np.ndarray | None]],
    row_labels: list[str],
    path: Path,
) -> None:
    """A grid of images with a title above each column and a label beside each row.

    Args:
        column_titles: One title per column.
        rows: For each row, one image (H, W, 3) per column, or ``None`` for an empty cell.
        row_labels: One label per row.
        path: Where to save the PNG.
    """
    cell_size = 1.5
    with matplotlib.rc_context(_STYLE):
        figure = Figure(
            figsize=(cell_size * len(column_titles) + 0.6, cell_size * len(rows) + 0.4),
            layout="constrained",
            dpi=_DPI,
        )
        axes = figure.subplots(len(rows), len(column_titles), squeeze=False)
        for row_index, (row, row_label) in enumerate(zip(rows, row_labels)):
            for column_index, image in enumerate(row):
                ax = axes[row_index, column_index]
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if image is not None:
                    ax.imshow(image)
                if row_index == 0:
                    ax.set_title(column_titles[column_index], fontsize=8, loc="center")
                if column_index == 0:
                    ax.set_ylabel(row_label, fontsize=7, color=_INK_MUTED)
        figure.savefig(path)


def _add_legend(figure: Figure, styles: dict[str, RunStyle], run_names: list[str], with_lines: bool = True) -> None:
    """One legend for the whole figure, above the panels, showing each run's colour, marker and line."""
    if len(run_names) < 2:
        return  # a single run needs no legend; the panel titles say what is shown
    handles = [
        Line2D(
            [],
            [],
            color=styles[name].color,
            marker=styles[name].marker,
            linestyle=styles[name].linestyle if with_lines else "none",
        )
        for name in run_names
    ]
    figure.legend(handles, run_names, loc="outside upper center", ncol=min(len(run_names), 4))


def _moving_average(steps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a noisy curve with a window of about a fiftieth of its length."""
    window = max(1, len(values) // 50)
    if window == 1:
        return steps, values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    return steps[window - 1 :], smoothed
