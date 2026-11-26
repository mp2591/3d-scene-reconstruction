"""Loading the training batches planned by a ``TrainingSchedule``.

The schedule decides what every step loads; this module only loads it,
possibly in background worker processes. Because the decisions are made in
the main process, the number of workers never changes the data.
"""

from __future__ import annotations

import mmap
import shutil
import sys
from itertools import count
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from scene_recon.data.base import SceneDataset
from scene_recon.data.schedule import TrainingSchedule
from scene_recon.structures import SceneViews

BATCHES_AHEAD_PER_WORKER = 1
"""Batches each worker process loads before training asks for them (PyTorch's ``prefetch_factor``).

One is enough: a worker loads a batch far faster than the GPU trains on one
(a batch of SRN cars took 22 ms to load on one core of the RTX 4090
machine, a training step 240 ms or more), so with several workers the next
batch is always ready. Fewer batches waiting also take less of the
shared memory below.
"""

SHARED_MEMORY = Path("/dev/shm")
"""Where worker processes on Linux put the batches they hand to the training process."""


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
    if num_workers > 0:
        check_shared_memory(schedule, start_step, num_workers)
    loader = DataLoader(
        _PlannedViews(schedule.scenes),
        batch_sampler=_StepPlans(schedule, start_step),
        num_workers=num_workers,
        prefetch_factor=BATCHES_AHEAD_PER_WORKER if num_workers > 0 else None,
        collate_fn=SceneViews.concatenate,
        # Without its own generator, the loader would draw a seed from
        # PyTorch's global generator, which belongs to the model: a resumed
        # run would then continue the model's randomness (dropout) differently.
        generator=torch.Generator().manual_seed(schedule.seed),
    )
    yield from loader


def check_shared_memory(
    schedule: TrainingSchedule, start_step: int, num_workers: int, folder: Path = SHARED_MEMORY
) -> None:
    """Refuse to start worker processes that would not fit their batches into shared memory.

    On Linux, a worker process hands each batch to the training process
    through ``/dev/shm``. It holds the batches loaded ahead, the one being
    handed over, and the loader's locks. Containers often get little of it,
    64 MB by default under Docker and Kubernetes, and a worker that finds
    it full stops training with a bus error, possibly hours in. The need is
    worked out from the batch of ``start_step``: every batch of a schedule
    has the same number of scenes and views. Each tensor takes whole memory
    pages there (64 KB each on the GH200's processor, 4 KB on most others).
    Nothing is checked on other systems, which share memory another way.
    """
    if not _hands_batches_over_in_shared_memory():
        return
    remedy = (
        "Give the machine more shared memory (Run:ai: --large-shm, or 'Increase shared memory size' in the "
        "workload's settings; Docker: --shm-size=1g), or lower training.num_workers in configs/base.yaml, which "
        "does not change what is trained: 0 loads every batch in the training process, more slowly."
    )
    if not folder.is_dir():
        raise RuntimeError(
            f"Training loads its batches in {num_workers} worker processes (training.num_workers), which hand "
            f"them over through {folder}, but this machine has no {folder}. {remedy}"
        )
    planned = _PlannedViews(schedule.scenes)
    batch = SceneViews.concatenate([planned[scene_plan] for scene_plan in schedule.scene_plan(start_step)])
    batches = num_workers * BATCHES_AHEAD_PER_WORKER + 1
    needed = batches * _shared_bytes(batch) + _loader_locks(num_workers) * mmap.PAGESIZE
    free = shutil.disk_usage(folder).free
    if free < needed:
        raise RuntimeError(
            f"Training loads its batches in {num_workers} worker processes (training.num_workers), which hand "
            f"them over through {folder}. That needs about {needed / 1e6:.1f} MB there, but only "
            f"{free / 1e6:.1f} MB is free, and a worker that finds it full stops training. {remedy}"
        )


def _hands_batches_over_in_shared_memory() -> bool:
    """Whether worker processes hand batches over through files in ``/dev/shm``: on Linux."""
    return sys.platform.startswith("linux")


def _shared_bytes(batch: SceneViews) -> int:
    """The shared memory a batch takes: each of its tensors in whole memory pages."""
    tensors = [batch.images, batch.cameras.camera_to_world, batch.cameras.intrinsics, batch.near, batch.far]
    pages = [-(-tensor.element_size() * tensor.nelement() // mmap.PAGESIZE) for tensor in tensors]
    return sum(pages) * mmap.PAGESIZE


def _loader_locks(num_workers: int) -> int:
    """The locks PyTorch's loader makes for its workers, each a page in shared memory.

    Three for each worker's queue of orders, three for the queue of results
    and five for the event that stops the workers.
    """
    return 3 * num_workers + 8
