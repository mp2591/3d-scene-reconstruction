"""Where each output of a run is stored.

Every run writes into its own directory, ``<experiment.output_dir>/<experiment.run_name>/``::

    config.yaml                      the complete config of the run
    checkpoints/last.pt              model, optimiser and scheduler state
    checkpoints/last.json            training progress (step, hours), readable without loading the model
    tensorboard/                     training curves and validation renders
    evaluation.json                  test metrics                          (written by `evaluate`)
    previews/<n>_source_views/       first rendered view of a few scenes   (written by `evaluate`)
    renders/<n>_source_views/        every rendered view, if enabled       (written by `evaluate`)
    profile.json                     parameter counts and rendering speed  (written by `profile`)

This module is the only place that knows these names. Everything that reads
or writes a run's outputs goes through ``RunFiles``, and results files are
written with ``write_json``, which never leaves a half-written file behind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RunFiles:
    """The paths of all outputs of one run."""

    run_dir: Path

    @property
    def config(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / "checkpoints" / "last.pt"

    @property
    def tensorboard(self) -> Path:
        return self.run_dir / "tensorboard"

    @property
    def evaluation(self) -> Path:
        return self.run_dir / "evaluation.json"

    @property
    def profile(self) -> Path:
        return self.run_dir / "profile.json"

    def previews(self, num_source_views: int) -> Path:
        """Directory of the preview renders of one evaluation protocol."""
        return self.run_dir / "previews" / f"{num_source_views}_source_views"

    def renders(self, num_source_views: int) -> Path:
        """Directory of all renders of one evaluation protocol (only with ``evaluation.save_images``)."""
        return self.run_dir / "renders" / f"{num_source_views}_source_views"


@dataclass(frozen=True)
class SuiteFiles:
    """The paths of a suite's own outputs, which sit next to its run directories::

        <output_dir>/<suite_name>/
            suite.log, summary.md, summary.csv, paired_comparison.csv, plots/
            <run_name>/ ...                                  (one directory per run)

    Because runs share this directory, their names must not be one of these;
    see ``reserved_names``.
    """

    directory: Path

    @property
    def log(self) -> Path:
        return self.directory / "suite.log"

    @property
    def summary_markdown(self) -> Path:
        return self.directory / "summary.md"

    @property
    def summary_csv(self) -> Path:
        return self.directory / "summary.csv"

    @property
    def paired_comparison_csv(self) -> Path:
        return self.directory / "paired_comparison.csv"

    @property
    def plots(self) -> Path:
        return self.directory / "plots"

    @staticmethod
    def reserved_names() -> set[str]:
        """Lower-case names a run may not have: the suite's outputs and their temporary files."""
        files = SuiteFiles(Path("."))
        file_names = [
            path.name for path in (files.log, files.summary_markdown, files.summary_csv, files.paired_comparison_csv)
        ]
        names = set(file_names)
        return {name.lower() for name in names} | {f"{name}.tmp".lower() for name in names} | {files.plots.name}


def write_atomically(path: Path, write: Callable[[Path], Any]) -> None:
    """Write a file through a temporary file that is then moved into place.

    A crash while writing therefore never leaves a truncated file behind:
    the path holds either the complete old content or the complete new one.

    Args:
        path: The file to write.
        write: Writes the content to the temporary path it is given.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    write(temporary_path)
    temporary_path.replace(path)


def write_json(content: dict[str, Any], path: Path) -> None:
    """Write a results file atomically."""
    write_atomically(path, lambda temporary_path: temporary_path.write_text(json.dumps(content, indent=2)))


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a results file, or return ``None`` if it is missing or damaged.

    A damaged file (e.g. from a crash under an older version that wrote files
    directly) is treated like a missing one, so the work is simply redone.
    """
    try:
        content = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return content if isinstance(content, dict) else None
