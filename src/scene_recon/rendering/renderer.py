"""Rendering rays and whole images by querying a model along rays."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from scene_recon.config import RendererSettings
from scene_recon.models.base import LEVEL_COARSE, LEVEL_FINE, ReconstructionModel
from scene_recon.rendering.rays import rays_for_full_images
from scene_recon.rendering.sampling import sample_around_depth, sample_importance_depths, sample_stratified_depths
from scene_recon.rendering.volume import composite_along_rays
from scene_recon.structures import RayBundle, RenderOutput, SceneViews


class VolumeRenderer:
    """Renders rays with a coarse pass and an optional fine pass.

    The renderer has no learnable parameters. It decides where to sample,
    asks the model for colour and density there, and integrates the answers.
    """

    def __init__(self, settings: RendererSettings) -> None:
        self.settings = settings

    def render_rays(
        self,
        model: ReconstructionModel,
        scene_encoding: Any,
        rays: RayBundle,
        generator: torch.Generator | None = None,
    ) -> dict[str, RenderOutput]:
        """Render a set of rays.

        Args:
            model: The model to query.
            scene_encoding: The result of ``model.encode_scene`` for the same scenes as ``rays``.
            rays: The rays to render.
            generator: Jitter the sample positions with random numbers from
                this CPU generator (training and evaluation, as in pixelNeRF).
                ``None`` places them at fixed positions.

        Returns:
            One ``RenderOutput`` per rendering pass, keyed by level name:
            always ``"coarse"``, and ``"fine"`` if the fine pass is enabled.
        """
        background_color = torch.tensor(self.settings.background_color, device=rays.origins.device)

        coarse_depths = sample_stratified_depths(rays, self.settings.num_coarse_samples, generator)
        outputs = {
            LEVEL_COARSE: self._render_at_depths(
                model, scene_encoding, rays, coarse_depths, LEVEL_COARSE, background_color
            )
        }

        if self.settings.num_fine_samples > 0:
            # pixelNeRF's fine pass: num_fine_depth_samples of the fine samples
            # lie close to the depth the coarse pass found, the rest where it
            # found content. It reuses the coarse depths, so it never sees
            # less of the ray than the coarse pass did.
            coarse = outputs[LEVEL_COARSE]
            num_importance_samples = self.settings.num_fine_samples - self.settings.num_fine_depth_samples
            depth_sets = [coarse_depths]
            if num_importance_samples > 0:
                depth_sets.append(sample_importance_depths(rays, coarse.weights, num_importance_samples, generator))
            if self.settings.num_fine_depth_samples > 0:
                depth_sets.append(
                    sample_around_depth(
                        rays, coarse.depth, self.settings.num_fine_depth_samples, self.settings.fine_depth_std, generator
                    )
                )
            fine_depths, _ = torch.sort(torch.cat(depth_sets, dim=-1), dim=-1)
            outputs[LEVEL_FINE] = self._render_at_depths(
                model, scene_encoding, rays, fine_depths, LEVEL_FINE, background_color
            )

        return outputs

    @torch.no_grad()
    def render_images(
        self,
        model: ReconstructionModel,
        scene_encoding: Any,
        target_views: SceneViews,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Render full images from the viewpoints of ``target_views``.

        Only the cameras, the image size and the near/far range of
        ``target_views`` are used; its images are never read. Rays are
        rendered in chunks of ``ray_chunk_size`` to bound memory use.
        With a ``generator``, the sample positions are random, drawn chunk
        by chunk, so which numbers a ray gets depends on the chunk size;
        without one, the images do not depend on it.

        Returns:
            ``(rgb, depth)`` with shapes (B, N, 3, H, W) and (B, N, H, W).
        """
        rays = rays_for_full_images(target_views)
        rgb_chunks, depth_chunks = [], []
        for start in range(0, rays.num_rays, self.settings.ray_chunk_size):
            chunk = rays.slice(start, start + self.settings.ray_chunk_size)
            output = final_output(self.render_rays(model, scene_encoding, chunk, generator))
            rgb_chunks.append(output.rgb)
            depth_chunks.append(output.depth)

        batch_size, num_views = target_views.num_scenes, target_views.num_views
        height, width = target_views.image_size
        rgb = torch.cat(rgb_chunks, dim=1).reshape(batch_size, num_views, height, width, 3)
        depth = torch.cat(depth_chunks, dim=1).reshape(batch_size, num_views, height, width)
        return rgb.permute(0, 1, 4, 2, 3), depth

    @staticmethod
    def _render_at_depths(
        model: ReconstructionModel,
        scene_encoding: Any,
        rays: RayBundle,
        depths: Tensor,
        level: str,
        background_color: Tensor,
    ) -> RenderOutput:
        """Query the model at the given depths along each ray and composite the result."""
        points = rays.origins[:, :, None] + depths[..., None] * rays.directions[:, :, None]
        field = model.query_field(scene_encoding, points, rays.directions, level)
        return composite_along_rays(field, depths, rays.far, background_color)


def final_output(outputs: dict[str, RenderOutput]) -> RenderOutput:
    """The most refined render available: the fine pass if it ran, otherwise the coarse pass."""
    return outputs.get(LEVEL_FINE, outputs[LEVEL_COARSE])
