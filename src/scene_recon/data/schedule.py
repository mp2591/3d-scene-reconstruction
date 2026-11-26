"""The training data schedule: exactly which data every training step uses.

For a fair comparison, every model must be trained on the same data in the
same order. This module makes every random choice about the training data a
function of only two numbers, the experiment's seed and the step number:

    which scenes are in the batch        scene_plan()
    which views of them are loaded       scene_plan()
    which views the model is given       source_view_indices()
    whether rays come from the object    rays_in_object_box()
    which pixels are supervised          generator("rays")
    where along each ray samples go      generator("sample_depths")

Nothing the model does can change these choices, so:

* every model with the same seed sees identical training data at every step,
  however it initialises its weights or uses randomness (dropout, stochastic depth);
* a run resumed from a checkpoint continues with exactly the data it would
  have seen without the interruption;
* the number of data-loading workers, and the device, make no difference.

Each choice uses its own random generator, seeded from (seed, purpose, step).
Generators live on the CPU; their results are moved to the device afterwards,
so a CPU run and a GPU run draw the same numbers.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch

from scene_recon.data.base import SceneDataset
from scene_recon.registry import ConfigError

if TYPE_CHECKING:
    from scene_recon.config import TrainingSettings


def derived_generator(seed: int, purpose: str, *keys: int) -> torch.Generator:
    """A random generator determined by the seed, a purpose and any number of integer keys.

    Different purposes or keys give unrelated streams of random numbers, so
    adding a new random choice somewhere never changes the existing ones.
    """
    text = "/".join([str(seed), purpose, *(str(key) for key in keys)])
    derived_seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return torch.Generator().manual_seed(derived_seed)


class TrainingSchedule:
    """Decides the training data of every step from the seed and the step number."""

    def __init__(self, scenes: SceneDataset, seed: int, settings: TrainingSettings) -> None:
        """
        Args:
            scenes: The training scenes.
            seed: The experiment's seed.
            settings: The config's training section: batch_size, views_per_scene,
                num_source_views and the first_phase settings are used.
        """
        self.scenes = scenes
        self.seed = seed
        self.batch_size = batch_size = settings.batch_size
        self.views_per_scene = settings.views_per_scene
        self.num_source_views = settings.num_source_views
        self.first_phase_steps = settings.first_phase_steps
        self.first_phase_num_source_views = settings.first_phase_num_source_views
        self.first_phase_rays_in_object_box = settings.first_phase_rays_in_object_box

        num_scenes = len(scenes)
        # Every pass over the data (epoch) visits each scene at most once. A
        # final incomplete batch is dropped so every step has the same size,
        # unless there are fewer scenes than batch_size; then each step uses all of them.
        self.batches_per_epoch = max(1, num_scenes // batch_size)
        self.scenes_per_step = min(batch_size, num_scenes)

    def scene_plan(self, step: int) -> list[tuple[int, list[int]]]:
        """Which scenes, and which of their views, step ``step`` loads.

        Returns:
            One ``(scene_index, view_indices)`` pair per scene of the batch.
        """
        epoch, position = divmod(step, self.batches_per_epoch)
        scene_order = torch.randperm(len(self.scenes), generator=derived_generator(self.seed, "scene_order", epoch))
        start = position * self.scenes_per_step
        scene_indices = scene_order[start : start + self.scenes_per_step].tolist()

        plan = []
        for item, scene_index in enumerate(scene_indices):
            num_available = self.scenes.num_views(scene_index)
            if num_available < self.views_per_scene:
                raise ConfigError(
                    f"Training scene {scene_index} has {num_available} views, but "
                    f"training.views_per_scene is {self.views_per_scene}."
                )
            views = torch.randperm(num_available, generator=derived_generator(self.seed, "views", step, item))
            plan.append((scene_index, views[: self.views_per_scene].tolist()))
        return plan

    def source_view_indices(self, step: int) -> list[int]:
        """Which of the loaded views step ``step`` gives to the model as input.

        The number of source views is drawn every step, from
        ``first_phase_num_source_views`` during the first phase and from
        ``num_source_views`` after it, so one model learns to work with any of them.
        """
        counts = self.first_phase_num_source_views if self.in_first_phase(step) else self.num_source_views
        generator = derived_generator(self.seed, "source_views", step)
        choice = torch.randint(len(counts), (1,), generator=generator).item()
        count = counts[choice]
        return torch.randperm(self.views_per_scene, generator=generator)[:count].tolist()

    def in_first_phase(self, step: int) -> bool:
        """Whether step ``step`` (counted from 0) belongs to the first phase of training."""
        return step < self.first_phase_steps

    def rays_in_object_box(self, step: int) -> bool:
        """Whether step ``step`` draws its rays only from the box around each view's object."""
        return self.first_phase_rays_in_object_box and self.in_first_phase(step)

    def generator(self, purpose: str, step: int) -> torch.Generator:
        """A generator for one of the step's other random choices, e.g. ``"rays"``."""
        return derived_generator(self.seed, purpose, step)
