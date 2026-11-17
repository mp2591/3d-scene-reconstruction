"""Novel-view evaluation: render held-out views of each scene and score them.

For every scene the model is shown a fixed set of source views, renders the
remaining views in full, and each rendered image is compared with the ground
truth. Metrics are averaged over the views of a scene first and over scenes
second, so every scene counts equally regardless of how many views it has.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from scene_recon.data.base import SceneDataset
from scene_recon.models.base import ReconstructionModel
from scene_recon.registry import ConfigError
from scene_recon.rendering import VolumeRenderer

logger = logging.getLogger(__name__)

_LOG_PROGRESS_EVERY_NUM_SCENES = 25


@dataclass
class EvaluationResult:
    mean: dict[str, float]
    """Each metric averaged over all evaluated scenes."""

    per_scene: dict[str, dict[str, float]]
    """Each metric averaged over the target views of one scene, keyed by scene id."""

    preview: Tensor
    """An image of shape (3, H, 3 * W) for visual inspection, showing for the first scene:
    its first source view, the first rendered target view, and that view's ground truth."""


class Evaluator:
    """Scores a model on a dataset with a given set of metrics."""

    def __init__(
        self,
        model: ReconstructionModel,
        renderer: VolumeRenderer,
        metrics: dict[str, nn.Module],
        device: torch.device,
    ) -> None:
        self.model = model
        self.renderer = renderer
        self.metrics = {name: metric.to(device) for name, metric in metrics.items()}
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        scenes: SceneDataset,
        source_view_indices: Sequence[int],
        num_scenes: int | None = None,
        num_target_views: int | None = None,
        image_dir: Path | None = None,
    ) -> EvaluationResult:
        """Evaluate on the first ``num_scenes`` scenes of a dataset.

        Args:
            scenes: The dataset to evaluate on.
            source_view_indices: Views shown to the model, the same for every scene.
            num_scenes: Number of scenes to evaluate. ``None`` means all.
            num_target_views: Number of views to render per scene, spread
                evenly over the non-source views. ``None`` means all of them.
            image_dir: If given, every rendered image is saved as
                ``<image_dir>/<scene_id>/<view_index>.png``.
        """
        num_scenes = len(scenes) if num_scenes is None else min(num_scenes, len(scenes))
        source_view_indices = list(source_view_indices)
        num_source_views = len(source_view_indices)

        was_training = self.model.training
        self.model.eval()

        per_scene: dict[str, dict[str, float]] = {}
        preview = None
        for scene_index in range(num_scenes):
            target_view_indices = _choose_target_views(
                scenes.num_views(scene_index), source_view_indices, num_target_views
            )
            views = scenes.load_scene(scene_index, source_view_indices + target_view_indices).to(self.device)
            source_views = views.select_views(range(num_source_views))
            target_views = views.select_views(range(num_source_views, views.num_views))

            scene_encoding = self.model.encode_scene(source_views)
            rendered, _ = self.renderer.render_images(self.model, scene_encoding, target_views)

            # A single scene was loaded, so the batch dimension is dropped.
            prediction = rendered[0].clamp(0.0, 1.0)
            ground_truth = target_views.images[0]
            scene_id = views.scene_ids[0]

            per_scene[scene_id] = {
                name: metric(prediction, ground_truth).mean().item() for name, metric in self.metrics.items()
            }
            if preview is None:
                preview = torch.cat([source_views.images[0, 0], prediction[0], ground_truth[0]], dim=-1).cpu()
            if image_dir is not None:
                _save_images(prediction, target_view_indices, image_dir / scene_id)
            if (scene_index + 1) % _LOG_PROGRESS_EVERY_NUM_SCENES == 0:
                logger.info("Evaluated %d / %d scenes", scene_index + 1, num_scenes)

        self.model.train(was_training)

        mean = {
            name: float(np.mean([scene_metrics[name] for scene_metrics in per_scene.values()]))
            for name in self.metrics
        }
        return EvaluationResult(mean=mean, per_scene=per_scene, preview=preview)


def _choose_target_views(
    num_views: int, source_view_indices: list[int], num_target_views: int | None
) -> list[int]:
    """All views that are not source views, optionally thinned out to an evenly spaced subset."""
    if max(source_view_indices) >= num_views:
        raise ConfigError(
            f"Source view indices {source_view_indices} are out of range for a scene with "
            f"{num_views} views. Adjust the source views in the 'validation' or 'evaluation' config section."
        )
    candidates = [index for index in range(num_views) if index not in source_view_indices]
    if not candidates:
        raise ConfigError("Every view of the scene is a source view, so there is nothing left to evaluate on.")
    if num_target_views is None or num_target_views >= len(candidates):
        return candidates

    positions = np.linspace(0, len(candidates) - 1, num_target_views).round().astype(int)
    return [candidates[position] for position in positions]


def _save_images(images: Tensor, view_indices: list[int], directory: Path) -> None:
    """Save images of shape (N, 3, H, W) in [0, 1] as 8-bit PNG files named after their view index."""
    directory.mkdir(parents=True, exist_ok=True)
    pixels = (images.permute(0, 2, 3, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    for image, view_index in zip(pixels, view_indices):
        Image.fromarray(image).save(directory / f"{view_index:06d}.png")
