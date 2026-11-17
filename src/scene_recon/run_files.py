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
or writes a run's outputs goes through ``RunFiles``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
