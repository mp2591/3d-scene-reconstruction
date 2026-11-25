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
        ``encode_seconds``, ``seconds_per_frame``, ``frames_per_second``, and
        two GPU memory peaks while rendering frames, in GB (``None`` on the
        CPU): ``peak_memory_allocated_gb``, held by tensors, and
        ``peak_memory_reserved_gb``, held by PyTorch from the driver, which
        includes memory kept for reuse.

    Raises:
        RuntimeError: If rendering filled the GPU's memory. Some drivers (on
            Windows) then move memory to the computer's main memory instead
            of failing, which makes rendering several times slower, so the
            measured speed would be wrong.
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
    memory = _check_memory_fits(device, renderer)

    model.train(was_training)
    return {
        "encode_seconds": encode_seconds,
        "seconds_per_frame": seconds_per_frame,
        "frames_per_second": 1.0 / seconds_per_frame,
        "peak_memory_allocated_gb": memory["allocated"] / 1e9 if memory else None,
        "peak_memory_reserved_gb": memory["reserved"] / 1e9 if memory else None,
    }


def _start_counting_memory(device: torch.device) -> None:
    """Release memory PyTorch keeps for reuse, so that the peaks count rendering alone."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


MIN_FREE_GPU_MEMORY = 0.02
"""Share of the GPU's memory that must still be free after the timed frames.

Measured on an RTX 4090 (22.5 GB) under Windows, at 128x128 with two source
views: at 2048 rays per chunk PyTorch reserved 5.8 GB and 15.1 GB stayed
free; at 8192 it reserved 22.9 GB (tensors used only 13.9 GB of it), nothing
stayed free, the driver moved the rest to main memory and the frame rate fell
from 0.93 to 0.39. Memory used by other programs counts too, so the check
uses what the driver reports as free, not only what PyTorch reserved.
"""


def _check_memory_fits(device: torch.device, renderer: VolumeRenderer) -> dict[str, int] | None:
    """The GPU memory of the rendered frames (see ``gpu_memory``), after checking that it fitted on the GPU."""
    if device.type != "cuda":
        return None
    memory = gpu_memory(device)
    if memory["reserved"] > memory["total"] or memory["free"] < MIN_FREE_GPU_MEMORY * memory["total"]:
        raise RuntimeError(
            f"Rendering filled the memory of {device}: PyTorch reserved {memory['reserved'] / 1e9:.1f} GB of "
            f"{memory['total'] / 1e9:.1f} GB, and {memory['free'] / 1e9:.1f} GB stayed free. Part of it may have "
            "been moved to main memory, so the measured speed would be wrong. Lower renderer.ray_chunk_size "
            f"(now {renderer.settings.ray_chunk_size}) for every run of the suite, or close other programs "
            "that use the GPU."
        )
    return memory


def gpu_memory(device: torch.device) -> dict[str, int]:
    """In bytes: the most memory PyTorch held on a GPU since it was last reset, and what is left now.

    * ``allocated``: the peak held by tensors;
    * ``reserved``: the peak PyTorch held from the driver, including memory
      kept for reuse;
    * ``free``: what the driver reports as free now, after the other
      programs on the GPU and the memory PyTorch still holds;
    * ``total``: the GPU's memory.
    """
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated": torch.cuda.max_memory_allocated(device),
        "reserved": torch.cuda.max_memory_reserved(device),
        "free": free,
        "total": total,
    }


def _wait_for_gpu(device: torch.device) -> None:
    """Block until queued GPU work has finished.

    CUDA calls return before the work is done; without this, a timer would
    measure how long it takes to queue the work rather than to do it.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
