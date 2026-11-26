"""Choosing which rays to render."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from scene_recon.geometry import pixel_centers, rays_through_pixels
from scene_recon.structures import Cameras, RayBundle, SceneViews


def sample_training_rays(
    views: SceneViews,
    rays_per_scene: int,
    generator: torch.Generator,
    object_box_background: Sequence[float] | None = None,
) -> tuple[RayBundle, Tensor]:
    """Pick random pixels from random views and return their rays and colours.

    Rays are drawn uniformly from all pixels of all given views, or, with
    ``object_box_background``, uniformly from the smallest box around each
    view's object (see ``object_boxes``).

    Args:
        views: B scenes with N views each.
        rays_per_scene: Number of rays R to draw from each scene.
        generator: CPU generator the pixel choices are drawn from.
        object_box_background: The images' background colour (RGB in [0, 1]),
            to draw rays only from the box around the object; ``None`` for the whole image.

    Returns:
        The rays, and the ground-truth colours of their pixels with shape (B, R, 3).
    """
    batch_size, num_views = views.num_scenes, views.num_views
    height, width = views.image_size
    device = views.images.device

    shape = (batch_size, rays_per_scene)
    view_index = torch.randint(num_views, shape, generator=generator).to(device)
    scene_index = torch.arange(batch_size, device=device)[:, None]
    if object_box_background is None:
        pixel_x = torch.randint(width, shape, generator=generator).to(device)
        pixel_y = torch.randint(height, shape, generator=generator).to(device)
    else:
        (left, right), (top, bottom) = object_boxes(views.images, object_box_background)
        # The box of each ray's view, shape (B, R).
        left, right, top, bottom = (ends[scene_index, view_index] for ends in (left, right, top, bottom))
        pixel_x = left + (torch.rand(shape, generator=generator).to(device) * (right - left + 1)).long()
        pixel_y = top + (torch.rand(shape, generator=generator).to(device) * (bottom - top + 1)).long()
        pixel_x, pixel_y = torch.minimum(pixel_x, right), torch.minimum(pixel_y, bottom)  # rand < 1, but rounding

    cameras = Cameras(
        camera_to_world=views.cameras.camera_to_world[scene_index, view_index],
        intrinsics=views.cameras.intrinsics[scene_index, view_index],
    )
    pixels = torch.stack([pixel_x, pixel_y], dim=-1) + 0.5  # integer index -> pixel centre
    origins, directions = rays_through_pixels(cameras, pixels)

    target_rgb = views.images[scene_index, view_index, :, pixel_y, pixel_x]  # (B, R, 3)
    return RayBundle(origins, directions, views.near, views.far), target_rgb


OBJECT_TOLERANCE = 0.5 / 255
"""Half an 8-bit level: a channel at least one level away from the background's differs from it."""


def object_boxes(images: Tensor, background: Sequence[float]) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """The smallest box around the object in each image.

    A pixel belongs to the object if every one of its channels differs from
    the background colour. This is pixelNeRF's rule for SRN (a pixel is
    foreground if none of its channels is 255): a pale grey edge pixel
    counts, a pixel with one channel at the background's value does not.

    Args:
        images: Shape (B, N, 3, H, W), values in [0, 1].
        background: The background colour, RGB in [0, 1].

    Returns:
        ``((left, right), (top, bottom))``: the first and last column and row
        of each box (inclusive), each of shape (B, N). An image without an
        object gets the whole image as its box.
    """
    height, width = images.shape[-2:]
    colour = torch.tensor(background, dtype=images.dtype, device=images.device)[:, None, None]
    is_object = ((images - colour).abs() > OBJECT_TOLERANCE).all(dim=-3)  # (B, N, H, W)

    def first_and_last(has_object: Tensor, size: int) -> tuple[Tensor, Tensor]:
        found = has_object.any(dim=-1)
        first = torch.where(found, has_object.int().argmax(dim=-1), 0)
        last = torch.where(found, size - 1 - has_object.flip(-1).int().argmax(dim=-1), size - 1)
        return first, last

    return first_and_last(is_object.any(dim=-2), width), first_and_last(is_object.any(dim=-1), height)


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
