"""Choosing which rays to render."""

from __future__ import annotations

import torch
from torch import Tensor

from scene_recon.geometry import pixel_centers, rays_through_pixels
from scene_recon.structures import Cameras, RayBundle, SceneViews


def sample_training_rays(views: SceneViews, rays_per_scene: int) -> tuple[RayBundle, Tensor]:
    """Pick random pixels from random views and return their rays and colours.

    Rays are drawn uniformly from all pixels of all given views.

    Args:
        views: B scenes with N views each.
        rays_per_scene: Number of rays R to draw from each scene.

    Returns:
        The rays, and the ground-truth colours of their pixels with shape (B, R, 3).
    """
    batch_size, num_views = views.num_scenes, views.num_views
    height, width = views.image_size
    device = views.images.device

    view_index = torch.randint(num_views, (batch_size, rays_per_scene), device=device)
    pixel_x = torch.randint(width, (batch_size, rays_per_scene), device=device)
    pixel_y = torch.randint(height, (batch_size, rays_per_scene), device=device)
    scene_index = torch.arange(batch_size, device=device)[:, None]

    cameras = Cameras(
        camera_to_world=views.cameras.camera_to_world[scene_index, view_index],
        intrinsics=views.cameras.intrinsics[scene_index, view_index],
    )
    pixels = torch.stack([pixel_x, pixel_y], dim=-1) + 0.5  # integer index -> pixel centre
    origins, directions = rays_through_pixels(cameras, pixels)

    target_rgb = views.images[scene_index, view_index, :, pixel_y, pixel_x]  # (B, R, 3)
    return RayBundle(origins, directions, views.near, views.far), target_rgb


def rays_for_full_images(views: SceneViews) -> RayBundle:
    """Rays through every pixel of every view.

    Returns:
        A bundle with R = N * H * W rays per scene, ordered view by view and
        row by row within a view. Reshaping a per-ray result of shape
        (B, R, ...) to (B, N, H, W, ...) gives back images.
    """
    batch_size, num_views = views.num_scenes, views.num_views
    height, width = views.image_size
    pixels = pixel_centers(height, width, device=views.images.device)  # (H * W, 2)
    num_pixels = pixels.shape[0]

    cameras = Cameras(
        camera_to_world=views.cameras.camera_to_world[:, :, None].expand(-1, -1, num_pixels, -1, -1),
        intrinsics=views.cameras.intrinsics[:, :, None].expand(-1, -1, num_pixels, -1, -1),
    )
    origins, directions = rays_through_pixels(cameras, pixels.expand(batch_size, num_views, -1, -1))
    return RayBundle(origins.flatten(1, 2), directions.flatten(1, 2), views.near, views.far)
