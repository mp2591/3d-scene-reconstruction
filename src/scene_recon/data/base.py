"""The interface every dataset implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from scene_recon.structures import SceneViews


class SceneDataset(ABC):
    """A collection of scenes, each observed from several posed views.

    A dataset only knows how to load views. Which views are used as model
    input and which as supervision is decided later by the trainer and the
    evaluator, so the same dataset class serves training and evaluation.

    To add a dataset, subclass this, decorate the class with
    ``@DATASETS.register("some_name")`` and import the module in
    ``scene_recon/data/__init__.py``.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of scenes."""

    @abstractmethod
    def num_views(self, scene_index: int) -> int:
        """Number of views available for one scene."""

    @abstractmethod
    def load_scene(self, scene_index: int, view_indices: Sequence[int] | None = None) -> SceneViews:
        """Load some or all views of one scene.

        Args:
            scene_index: Which scene, in ``range(len(self))``.
            view_indices: Which views to load, in the given order. ``None``
                loads every view.

        Returns:
            A ``SceneViews`` holding a single scene (B = 1), on the CPU.
        """
