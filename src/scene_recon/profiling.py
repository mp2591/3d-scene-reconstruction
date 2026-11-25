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
        ``encode_seconds``, ``seconds_per_frame``, ``frames_per_second`` and,
        on a GPU, ``peak_memory_gb``: the most GPU memory in use while
        rendering a frame (``None`` on the CPU).

    Raises:
        RuntimeError: If rendering needed more memory than the GPU has. Some
            drivers (on Windows) then move memory to the computer's main
            memory instead of failing, which makes rendering many times
            slower, so the measured speed would be wrong.
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

    _start_counting_memory(device)
    for _ in range(num_warmup_frames):
        renderer.render_images(model, scene_encoding, target_view)
    _wait_for_gpu(device)
    start = time.perf_counter()
    for _ in range(num_frames):
        renderer.render_images(model, scene_encoding, target_view)
    _wait_for_gpu(device)
    seconds_per_frame = (time.perf_counter() - start) / num_frames
    peak_memory_gb = _check_memory_fits(device, renderer)

    model.train(was_training)
    return {
        "encode_seconds": encode_seconds,
        "seconds_per_frame": seconds_per_frame,
        "frames_per_second": 1.0 / seconds_per_frame,
        "peak_memory_gb": peak_memory_gb,
    }


def _start_counting_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _check_memory_fits(device: torch.device, renderer: VolumeRenderer) -> float | None:
    """The peak GPU memory of the rendered frames in GB, after checking it fitted on the GPU."""
    if device.type != "cuda":
        return None
    peak, total = gpu_memory_bytes(device)
    if peak > total:
        raise RuntimeError(
            f"Rendering a frame needed {peak / 1e9:.1f} GB, more than the {total / 1e9:.1f} GB of {device}, so "
            "the measured speed would be wrong. Lower renderer.ray_chunk_size (now "
            f"{renderer.settings.ray_chunk_size}) for every run of the suite."
        )
    return peak / 1e9


def gpu_memory_bytes(device: torch.device) -> tuple[int, int]:
    """The most memory PyTorch had in use on a GPU since it was last reset, and the GPU's total memory."""
    return torch.cuda.max_memory_allocated(device), torch.cuda.get_device_properties(device).total_memory


def _wait_for_gpu(device: torch.device) -> None:
    """Block until queued GPU work has finished.

    CUDA calls return before the work is done; without this, a timer would
    measure how long it takes to queue the work rather than to do it.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
