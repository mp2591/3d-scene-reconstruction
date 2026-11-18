"""Loading the training batches planned by a ``TrainingSchedule``.

The schedule decides what every step loads; this module only loads it,
possibly in background worker processes. Because the decisions are made in
the main process, the number of workers never changes the data.
"""

from __future__ import annotations

from itertools import count
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from scene_recon.data.base import SceneDataset
from scene_recon.data.schedule import TrainingSchedule
from scene_recon.structures import SceneViews


class _PlannedViews(Dataset):
    """Loads one ``(scene_index, view_indices)`` pair of a schedule's plan."""

    def __init__(self, scenes: SceneDataset) -> None:
        self.scenes = scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, planned: tuple[int, list[int]]) -> SceneViews:
        scene_index, view_indices = planned
        return self.scenes.load_scene(scene_index, view_indices)


class _StepPlans(Sampler):
    """Yields the scene plan of every step, from ``start_step`` on, forever."""

    def __init__(self, schedule: TrainingSchedule, start_step: int) -> None:
        self.schedule = schedule
        self.start_step = start_step

    def __iter__(self) -> Iterator[list[tuple[int, list[int]]]]:
        for step in count(self.start_step):
            yield self.schedule.scene_plan(step)


def training_batches(schedule: TrainingSchedule, start_step: int, num_workers: int) -> Iterator[SceneViews]:
    """Yield the batch of every training step, starting at ``start_step``.

    The n-th batch yielded is the batch of step ``start_step + n``.
    Training is counted in steps rather than epochs, so the trainer simply
    takes as many batches as it needs.
    """
    loader = DataLoader(
        _PlannedViews(schedule.scenes),
        batch_sampler=_StepPlans(schedule, start_step),
        num_workers=num_workers,
        collate_fn=SceneViews.concatenate,
        # Without its own generator, the loader would draw a seed from
        # PyTorch's global generator, which belongs to the model: a resumed
        # run would then continue the model's randomness (dropout) differently.
        generator=torch.Generator().manual_seed(schedule.seed),
    )
    yield from loader
