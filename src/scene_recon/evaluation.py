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

# File names inside a preview directory (see save_preview).
_PREVIEW_PREDICTION = "prediction.png"
_PREVIEW_TRUTH = "truth.png"
_PREVIEW_SOURCE_PATTERN = "source_*.png"


@dataclass
class ScenePreview:
    """One rendered view of one scene, kept for visual inspection.

    The view is always the first non-source view of the scene, so every
    model evaluated with the same source views renders the same image.
    """

    scene_id: str
    target_view: int
    """Index of the rendered view within the scene."""

    source_images: Tensor
    """The images the model was given, shape (V, 3, H, W)."""

    prediction: Tensor
    """The rendered view, shape (3, H, W)."""

    truth: Tensor
    """The true image of the same view, shape (3, H, W)."""

    def side_by_side(self) -> Tensor:
        """First source image, prediction and truth next to each other, shape (3, H, 3 * W)."""
        return torch.cat([self.source_images[0], self.prediction, self.truth], dim=-1)


@dataclass
class EvaluationResult:
    mean: dict[str, float]
    """Each metric averaged over all evaluated scenes."""

    per_scene: dict[str, dict[str, float]]
    """Each metric averaged over the target views of one scene, keyed by scene id."""

    previews: list[ScenePreview]
    """One preview for each of the first ``num_preview_scenes`` scenes."""


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
        num_preview_scenes: int = 1,
        image_dir: Path | None = None,
    ) -> EvaluationResult:
        """Evaluate on the first ``num_scenes`` scenes of a dataset.

        Args:
            scenes: The dataset to evaluate on.
            source_view_indices: Views shown to the model, the same for every scene.
            num_scenes: Number of scenes to evaluate. ``None`` means all.
            num_target_views: Number of views to render per scene, spread
                evenly over the non-source views. ``None`` means all of them.
            num_preview_scenes: Number of scenes, from the first one on, for
                which a ``ScenePreview`` is returned.
            image_dir: If given, every rendered image is saved as
                ``<image_dir>/<scene_id>/<view_index>.png``.
        """
        num_scenes = len(scenes) if num_scenes is None else min(num_scenes, len(scenes))
        source_view_indices = list(source_view_indices)
        num_source_views = len(source_view_indices)

        was_training = self.model.training
        self.model.eval()

        per_scene: dict[str, dict[str, float]] = {}
        previews: list[ScenePreview] = []
        for scene_index in range(num_scenes):
            target_view_indices = choose_target_views(
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
            if scene_index < num_preview_scenes:
                previews.append(
                    ScenePreview(
                        scene_id=scene_id,
                        target_view=target_view_indices[0],
                        source_images=source_views.images[0].cpu(),
                        prediction=prediction[0].cpu(),
                        truth=ground_truth[0].cpu(),
                    )
                )
            if image_dir is not None:
                for image, view_index in zip(prediction, target_view_indices):
                    save_image(image, image_dir / scene_id / f"{view_index:06d}.png")
            if (scene_index + 1) % _LOG_PROGRESS_EVERY_NUM_SCENES == 0:
                logger.info("Evaluated %d / %d scenes", scene_index + 1, num_scenes)

        self.model.train(was_training)

        mean = {
            name: float(np.mean([scene_metrics[name] for scene_metrics in per_scene.values()]))
            for name in self.metrics
        }
        return EvaluationResult(mean=mean, per_scene=per_scene, previews=previews)


def choose_target_views(
    num_views: int, source_view_indices: Sequence[int], num_target_views: int | None
) -> list[int]:
    """All views that are not source views, optionally thinned out to an evenly spaced subset.

    The first non-source view is always included, which is what makes
    previews of different runs show the same view.
    """
    check_source_views(source_view_indices, num_views, where="the evaluation source views")
    candidates = [index for index in range(num_views) if index not in source_view_indices]
    if not candidates:
        raise ConfigError("Every view of the scene is a source view, so there is nothing left to evaluate on.")
    if num_target_views is None or num_target_views >= len(candidates):
        return candidates

    positions = np.linspace(0, len(candidates) - 1, num_target_views).round().astype(int)
    return [candidates[position] for position in positions]


def check_protocol_feasible(
    scenes: SceneDataset, source_view_indices: Sequence[int], num_scenes: int | None, where: str
) -> None:
    """Check, before any training, that a protocol can run on every scene it will use.

    Each of those scenes must have the source views, and at least one other
    view to render and score. Scenes are checked one by one, because a
    dataset's scenes need not all have the same number of views.

    Args:
        scenes: The dataset the protocol runs on.
        source_view_indices: The views given to the model.
        num_scenes: How many scenes, from the first, the protocol uses. ``None`` means all.
        where: The setting being checked, for error messages.
    """
    count = len(scenes) if num_scenes is None else min(num_scenes, len(scenes))
    for scene_index in range(count):
        num_views = scenes.num_views(scene_index)
        check_source_views(source_view_indices, num_views, where=f"{where} (scene {scene_index})")
        if num_views <= len(set(source_view_indices)):
            raise ConfigError(
                f"{where} {list(source_view_indices)} uses every view of scene {scene_index} ({num_views} views) "
                "as a source view, so no view is left to render and score."
            )


def check_source_views(source_view_indices: Sequence[int], num_views: int, where: str) -> None:
    """Raise a readable error if source view indices do not exist in a scene with ``num_views`` views."""
    out_of_range = [index for index in source_view_indices if not 0 <= index < num_views]
    if out_of_range:
        raise ConfigError(
            f"{where} include {out_of_range}, but the scenes have only {num_views} views "
            f"(valid indices are 0 to {num_views - 1})."
        )


def save_image(image: Tensor, path: Path) -> None:
    """Save an image of shape (3, H, W) with values in [0, 1] as an 8-bit PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = (image.clamp(0.0, 1.0).permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(pixels).save(path)


def save_preview(preview: ScenePreview, directory: Path) -> None:
    """Save a preview's images as PNG files in ``directory``."""
    for index, image in enumerate(preview.source_images):
        save_image(image, directory / f"source_{index}.png")
    save_image(preview.prediction, directory / _PREVIEW_PREDICTION)
    save_image(preview.truth, directory / _PREVIEW_TRUTH)


def load_preview_images(directory: Path) -> dict[str, list[np.ndarray] | np.ndarray]:
    """Read back the images written by ``save_preview``.

    Returns:
        ``{"sources": [...], "prediction": ..., "truth": ...}`` with each image
        an array of shape (H, W, 3) and dtype uint8.
    """

    def read(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    return {
        "sources": [read(path) for path in sorted(directory.glob(_PREVIEW_SOURCE_PATTERN))],
        "prediction": read(directory / _PREVIEW_PREDICTION),
        "truth": read(directory / _PREVIEW_TRUTH),
    }
