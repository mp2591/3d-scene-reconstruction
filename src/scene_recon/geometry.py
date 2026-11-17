"""Camera geometry: casting rays out of cameras and projecting points back in.

The functions in the second half of this file (``world_to_camera`` onwards)
are not used by the pipeline itself. They are the building blocks a
pixel-aligned model needs to look up image features for a 3D point, and are
provided so that every model uses the same, tested conventions.

Pixel convention: the centre of the top-left pixel is at (0.5, 0.5) and the
image spans [0, W] x [0, H]. Pixel coordinates are always ordered (x, y).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from scene_recon.structures import Cameras

# Points closer to the camera plane than this are treated as being this far
# in front of it, which keeps the perspective division finite.
_MIN_PROJECTION_DEPTH = 1e-6


def pixel_centers(height: int, width: int, device: torch.device | str | None = None) -> Tensor:
    """Coordinates of every pixel centre, in row-major order.

    Returns:
        Tensor of shape (H * W, 2) holding (x, y) pairs. Reshaping a per-pixel
        result of shape (H * W, ...) to (H, W, ...) gives back an image.
    """
    ys = torch.arange(height, dtype=torch.float32, device=device) + 0.5
    xs = torch.arange(width, dtype=torch.float32, device=device) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


def rays_through_pixels(cameras: Cameras, pixels: Tensor) -> tuple[Tensor, Tensor]:
    """Cast one ray per pixel coordinate.

    Args:
        cameras: Cameras with leading dimensions (...), one per ray.
        pixels: Pixel coordinates (x, y) of shape (..., 2).

    Returns:
        ``(origins, directions)``, both of shape (..., 3) in world
        coordinates. Directions have unit length.
    """
    focal = torch.stack([cameras.intrinsics[..., 0, 0], cameras.intrinsics[..., 1, 1]], dim=-1)
    principal_point = cameras.intrinsics[..., :2, 2]

    # Direction in the camera frame: the point on the z = 1 plane that
    # projects onto the pixel.
    xy_on_unit_plane = (pixels - principal_point) / focal
    directions_camera = torch.cat([xy_on_unit_plane, torch.ones_like(xy_on_unit_plane[..., :1])], dim=-1)

    rotation = cameras.camera_to_world[..., :3, :3]
    directions = (rotation @ directions_camera.unsqueeze(-1)).squeeze(-1)
    directions = F.normalize(directions, dim=-1)
    origins = cameras.camera_to_world[..., :3, 3].expand_as(directions)
    return origins, directions


def look_at_pose(camera_position: Tensor, target: Tensor, world_up: Tensor) -> Tensor:
    """Build a camera-to-world pose for a camera that looks at a target point.

    Args:
        camera_position: Shape (3,).
        target: Shape (3,). The point the camera's +z axis passes through.
        world_up: Shape (3,). The direction that should appear "up" in the image.

    Returns:
        Pose of shape (4, 4).
    """
    forward = F.normalize(target - camera_position, dim=-1)
    right = F.normalize(torch.linalg.cross(forward, world_up), dim=-1)
    down = torch.linalg.cross(forward, right)

    pose = torch.eye(4, dtype=camera_position.dtype)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = camera_position
    return pose


def world_to_camera(points_world: Tensor, camera_to_world: Tensor) -> Tensor:
    """Express world-space points in the frame of each of V cameras.

    Args:
        points_world: Shape (B, P, 3).
        camera_to_world: Shape (B, V, 4, 4).

    Returns:
        Shape (B, V, P, 3).
    """
    rotation = camera_to_world[..., :3, :3]  # (B, V, 3, 3)
    translation = camera_to_world[..., :3, 3]  # (B, V, 3)

    # The inverse of a rigid transform [R | t] maps x to R^T (x - t).
    offset = points_world[:, None] - translation[:, :, None]  # (B, V, P, 3)
    return torch.einsum("bvji,bvpj->bvpi", rotation, offset)


def rotate_directions_to_camera(directions_world: Tensor, camera_to_world: Tensor) -> Tensor:
    """Express world-space directions in the frame of each of V cameras.

    Args:
        directions_world: Shape (B, P, 3).
        camera_to_world: Shape (B, V, 4, 4).

    Returns:
        Shape (B, V, P, 3).
    """
    rotation = camera_to_world[..., :3, :3]
    return torch.einsum("bvji,bpj->bvpi", rotation, directions_world)


def project_to_pixels(points_camera: Tensor, intrinsics: Tensor) -> Tensor:
    """Project camera-space points onto the image plane.

    Points behind the camera have no meaningful projection; they are clamped
    to lie just in front of it, so the result is always finite. Compare
    ``points_camera[..., 2]`` with zero to find them.

    Args:
        points_camera: Shape (B, V, P, 3).
        intrinsics: Shape (B, V, 3, 3).

    Returns:
        Pixel coordinates (x, y) of shape (B, V, P, 2).
    """
    depth = points_camera[..., 2:3].clamp(min=_MIN_PROJECTION_DEPTH)
    focal = torch.stack([intrinsics[..., 0, 0], intrinsics[..., 1, 1]], dim=-1)[:, :, None]
    principal_point = intrinsics[..., :2, 2][:, :, None]
    return points_camera[..., :2] / depth * focal + principal_point


def sample_features_at_pixels(feature_maps: Tensor, pixels: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Bilinearly look up a feature vector for each pixel coordinate.

    The feature maps may have any resolution: pixel coordinates are given in
    the coordinate system of the original images and rescaled internally.
    Coordinates outside the image take the feature of the nearest border.

    Args:
        feature_maps: Shape (B, V, C, feature_height, feature_width).
        pixels: Pixel coordinates (x, y) of shape (B, V, P, 2), in image pixels.
        image_size: (height, width) of the images the pixels refer to.

    Returns:
        Shape (B, V, P, C).
    """
    batch_size, num_views, num_channels = feature_maps.shape[:3]
    num_points = pixels.shape[2]
    height, width = image_size

    # grid_sample expects coordinates in [-1, 1], where -1 and +1 are the
    # outer edges of the image (align_corners=False).
    image_extent = torch.tensor([width, height], dtype=pixels.dtype, device=pixels.device)
    normalized = pixels / image_extent * 2.0 - 1.0

    sampled = F.grid_sample(
        feature_maps.flatten(0, 1),
        normalized.reshape(batch_size * num_views, num_points, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )  # (B * V, C, P, 1)
    return sampled.reshape(batch_size, num_views, num_channels, num_points).transpose(2, 3)
