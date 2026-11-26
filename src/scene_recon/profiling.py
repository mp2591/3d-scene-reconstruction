"""Measuring the computational cost of a model: parameter counts and rendering speed.

The third cost figure of the proposal, training GPU hours, is recorded by the
trainer itself (see ``TrainingProgress.training_seconds``).
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

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
    num_warmup_frames: int,
    min_free_gpu_memory: float,
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
        min_free_gpu_memory: Share of the GPU's memory that must stay free
            at every reading (``profiling.min_free_gpu_memory``).

    Returns:
        ``encode_seconds``, ``seconds_per_frame``, ``frames_per_second``, and,
        in GB (``None`` on the CPU): ``peak_memory_allocated_gb``, the most
        memory held by tensors while rendering; ``peak_memory_reserved_gb``,
        the most PyTorch held from the driver, including memory kept for
        reuse; and ``least_free_memory_gb``, the least the driver had left
        at a reading: after each frame was queued, and once all had run.

    Raises:
        RuntimeError: If rendering filled the GPU's memory. Some drivers (on
            Windows) then move memory to the computer's main memory instead
            of failing, which makes rendering several times slower, so the
            measured speed would be wrong.
    """
    modes = {module: module.training for module in model.modules()}  # a part may be frozen in eval mode
    order = _parents_first(model)  # taken now: in evaluation mode, a part may add temporary parts of its own
    try:
        model.eval()  # inside the try: if a part fails to switch, the parts already switched are restored
        device = source_views.images.device
        scene_encoding = model.encode_scene(source_views)  # warm-up for the encoder
        _wait_for_gpu(device)
        start = time.perf_counter()
        scene_encoding = model.encode_scene(source_views)
        _wait_for_gpu(device)
        encode_seconds = time.perf_counter() - start

        # The driver's free memory is read after each frame is queued (a
        # fraction of a millisecond) and once all frames have run. PyTorch
        # takes memory when work is queued, so these readings see what it
        # holds; memory another program takes and releases between two
        # readings is missed.
        _start_counting_memory(device)
        least_free = None
        # Random sample positions, as evaluation draws them, so that the time
        # to draw them is counted too; which numbers they are does not matter.
        generator = torch.Generator().manual_seed(0)
        for _ in range(num_warmup_frames):
            renderer.render_images(model, scene_encoding, target_view, generator)
            least_free = _least(least_free, free_gpu_memory(device))
        _wait_for_gpu(device)
        start = time.perf_counter()
        for _ in range(num_frames):
            renderer.render_images(model, scene_encoding, target_view, generator)
            least_free = _least(least_free, free_gpu_memory(device))
        _wait_for_gpu(device)
        seconds_per_frame = (time.perf_counter() - start) / num_frames
        least_free = _least(least_free, free_gpu_memory(device))
        memory = _check_memory_fits(device, renderer, least_free, min_free_gpu_memory)
    finally:
        _restore_modes(order, modes)
    return {
        "encode_seconds": encode_seconds,
        "seconds_per_frame": seconds_per_frame,
        "frames_per_second": 1.0 / seconds_per_frame,
        "peak_memory_allocated_gb": memory["allocated"] / 1e9 if memory else None,
        "peak_memory_reserved_gb": memory["reserved"] / 1e9 if memory else None,
        "least_free_memory_gb": memory["least_free"] / 1e9 if memory else None,
    }


def _restore_modes(order: list[nn.Module], modes: dict[nn.Module, bool]) -> None:
    """Put every part of a model back into the training or evaluation mode it had.

    Each part is switched through its own ``train()``, so that a part whose
    ``train()`` does more than set its flag (an adapter that merges its
    weights for evaluation and splits them for training) is left in a
    consistent state. ``train()`` also switches every part below, so parents
    go first: ``order`` lists each part after every part that contains it
    (``_parents_first``), including a part shared by two parents. Both
    ``order`` and ``modes`` are taken before the model leaves its modes.
    """
    for module in order:
        if module.training != modes[module]:
            module.train(modes[module])
    for module, training in modes.items():
        module.training = training  # a part whose train() leaves its own flag alone


def _parents_first(model: nn.Module) -> list[nn.Module]:
    """Every part of the model, each one listed after all parts that contain it."""
    finished: list[nn.Module] = []
    visited: set[nn.Module] = set()

    def visit(module: nn.Module) -> None:
        visited.add(module)
        for child in module.children():
            if child not in visited:
                visit(child)
        finished.append(module)  # after everything it contains

    visit(model)
    return finished[::-1]


def _start_counting_memory(device: torch.device) -> None:
    """Release memory PyTorch keeps for reuse, so that the peaks count rendering alone."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _least(least: int | None, value: int | None) -> int | None:
    """The smaller of two memory readings; ``None`` stands for no reading (on the CPU, or before the first)."""
    if least is None:
        return value
    if value is None:
        return least
    return min(least, value)


def _check_memory_fits(
    device: torch.device, renderer: VolumeRenderer, least_free: int, min_free_gpu_memory: float
) -> dict[str, int] | None:
    """The GPU memory of the rendered frames, after checking that they fitted on the GPU.

    Returns:
        ``gpu_memory(device)`` with ``least_free``, the least free memory
        at a reading, in bytes; ``None`` on the CPU.
    """
    if device.type != "cuda":
        return None
    memory = {**gpu_memory(device), "least_free": least_free}
    if memory["reserved"] > memory["total"] or least_free < min_free_gpu_memory * memory["total"]:
        raise RuntimeError(
            f"Rendering filled the memory of {device}: PyTorch reserved up to {memory['reserved'] / 1e9:.1f} GB "
            f"of {memory['total'] / 1e9:.1f} GB, and at one point only {least_free / 1e9:.1f} GB was free "
            f"(profiling.min_free_gpu_memory asks for {min_free_gpu_memory:.0%}). Part of it may have been "
            "moved to main memory, so the measured speed would be wrong. Lower renderer.ray_chunk_size "
            f"(now {renderer.settings.ray_chunk_size}) for every run of the suite, or close other programs "
            "that use the GPU."
        )
    return memory


def gpu_memory(device: torch.device) -> dict[str, int]:
    """In bytes: the most memory PyTorch held on a GPU since it was last reset, and the GPU's memory.

    * ``allocated``: the peak held by tensors;
    * ``reserved``: the peak PyTorch held from the driver, including memory
      kept for reuse;
    * ``total``: the GPU's memory.
    """
    return {
        "allocated": torch.cuda.max_memory_allocated(device),
        "reserved": torch.cuda.max_memory_reserved(device),
        "total": torch.cuda.mem_get_info(device)[1],
    }


def free_gpu_memory(device: torch.device) -> int | None:
    """What the driver reports as free on a GPU now, in bytes, after other programs and PyTorch; ``None`` on the CPU."""
    return torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None


def _wait_for_gpu(device: torch.device) -> None:
    """Block until queued GPU work has finished.

    CUDA calls return before the work is done; without this, a timer would
    measure how long it takes to queue the work rather than to do it.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
