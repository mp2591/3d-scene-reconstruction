"""Measuring the computational cost of a model: parameter counts and rendering speed.

The third cost figure of the proposal, training GPU hours, is recorded by the
trainer itself (see ``TrainingProgress.training_seconds``).
"""

from __future__ import annotations

import time
from typing import Any

import torch

from scene_recon.models.base import ReconstructionModel
from scene_recon.rendering import VolumeRenderer
from scene_recon.structures import SceneViews


def count_parameters(model: ReconstructionModel) -> dict[str, Any]:
    """Count the model's parameters in total and per parameter group."""
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "groups": {
            group.name: sum(parameter.numel() for parameter in group.parameters)
            for group in model.parameter_groups()
        },
    }


@torch.no_grad()
def measure_rendering_speed(
    model: ReconstructionModel,
    renderer: VolumeRenderer,
    source_views: SceneViews,
    target_view: SceneViews,
    num_frames: int,
    num_warmup_frames: int = 2,
) -> dict[str, float]:
    """Time full-frame rendering of one scene.

    Encoding the source views and rendering a frame are timed separately,
    because an interactive viewer encodes a scene once and then renders many
    frames from it.

    Args:
        model: The model to profile, already on the device to measure on.
        renderer: The renderer, whose sample counts determine the cost per ray.
        source_views: One scene (B = 1) with the views the model is conditioned on.
        target_view: The same scene with exactly one view; its camera and
            image size define the frame that is rendered repeatedly.
        num_frames: Number of timed frames to average over.
        num_warmup_frames: Untimed frames rendered first, so that one-off
            costs (memory allocation, CUDA kernel selection) are not counted.

    Returns:
        ``encode_seconds``, ``seconds_per_frame`` and ``frames_per_second``.
    """
    was_training = model.training
    model.eval()
    device = source_views.images.device

    scene_encoding = model.encode_scene(source_views)  # warm-up for the encoder
    _wait_for_gpu(device)
    start = time.perf_counter()
    scene_encoding = model.encode_scene(source_views)
    _wait_for_gpu(device)
    encode_seconds = time.perf_counter() - start

    for _ in range(num_warmup_frames):
        renderer.render_images(model, scene_encoding, target_view)
    _wait_for_gpu(device)
    start = time.perf_counter()
    for _ in range(num_frames):
        renderer.render_images(model, scene_encoding, target_view)
    _wait_for_gpu(device)
    seconds_per_frame = (time.perf_counter() - start) / num_frames

    model.train(was_training)
    return {
        "encode_seconds": encode_seconds,
        "seconds_per_frame": seconds_per_frame,
        "frames_per_second": 1.0 / seconds_per_frame,
    }


def _wait_for_gpu(device: torch.device) -> None:
    """Block until queued GPU work has finished.

    CUDA calls return before the work is done; without this, a timer would
    measure how long it takes to queue the work rather than to do it.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
