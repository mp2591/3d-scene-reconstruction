"""Batching for training: turns a ``SceneDataset`` into a stream of scene batches."""

from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from scene_recon.data.base import SceneDataset
from scene_recon.structures import SceneViews


class RandomViewSubset(Dataset):
    """Wraps a ``SceneDataset`` so that each item is a random subset of a scene's views.

    Loading only a few views per scene keeps a training step cheap: a
    ShapeNet-SRN scene has 50 images, but a step needs only a handful of them.
    """

    def __init__(self, scenes: SceneDataset, views_per_scene: int) -> None:
        self.scenes = scenes
        self.views_per_scene = views_per_scene

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, scene_index: int) -> SceneViews:
        num_available = self.scenes.num_views(scene_index)
        if num_available < self.views_per_scene:
            raise ValueError(
                f"Scene {scene_index} has {num_available} views, but training.views_per_scene "
                f"is {self.views_per_scene}."
            )
        view_indices = torch.randperm(num_available)[: self.views_per_scene].tolist()
        return self.scenes.load_scene(scene_index, view_indices)


def endless_training_batches(
    scenes: SceneDataset, batch_size: int, views_per_scene: int, num_workers: int
) -> Iterator[SceneViews]:
    """Yield shuffled batches of scenes forever, reshuffling after each pass over the data.

    Training is counted in steps rather than epochs, so the trainer simply
    takes as many batches from this iterator as it needs.
    """
    loader = DataLoader(
        RandomViewSubset(scenes, views_per_scene),
        batch_size=batch_size,
        shuffle=True,
        # Dropping the last incomplete batch keeps the number of rays per
        # step constant. With fewer scenes than batch_size there is only an
        # incomplete batch, so it must be kept.
        drop_last=len(scenes) >= batch_size,
        num_workers=num_workers,
        collate_fn=SceneViews.concatenate,
        persistent_workers=num_workers > 0,
    )
    while True:
        yield from loader
