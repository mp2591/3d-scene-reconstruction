"""A procedurally generated dataset for tests and smoke runs.

Each scene is a single shaded sphere on a white background, rendered
analytically from cameras on the upper hemisphere. It needs no files on disk,
is fully deterministic, and is geometrically exact, which makes it a useful
sanity check: if a model cannot learn these scenes, the problem is in the
code and not in the data.

This dataset is not part of the experiments in the proposal.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from scene_recon.data.base import SceneDataset
from scene_recon.geometry import look_at_pose, pixel_centers, rays_through_pixels
from scene_recon.registry import DATASETS
from scene_recon.structures import Cameras, SceneViews

# The camera sits at this distance from the origin, like the SRN cars cameras.
_CAMERA_DISTANCE = 1.3
_NEAR = 0.8
_FAR = 1.8
_SPHERE_RADIUS_RANGE = (0.25, 0.4)
_FOCAL_LENGTH_IN_IMAGE_WIDTHS = 1.2


@DATASETS.register("synthetic_spheres")
class SyntheticSpheresDataset(SceneDataset):
    """Scenes that each contain one sphere of random size and colour."""

    def __init__(self, num_scenes: int, num_views: int, image_size: int, seed: int) -> None:
        """
        Args:
            num_scenes: How many distinct scenes the dataset contains.
            num_views: Views per scene.
            image_size: Height and width of the square images, in pixels.
            seed: Scenes are a deterministic function of the seed and their
                index. Use different seeds for the train, val and test splits.
        """
        self.num_scenes = num_scenes
        self.views_per_scene = num_views
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_scenes

    def fingerprint(self) -> dict:
        """The scenes are fully determined by the constructor's arguments."""
        return {
            "dataset": "synthetic_spheres",
            "num_scenes": self.num_scenes,
            "num_views": self.views_per_scene,
            "image_size": self.image_size,
            "seed": self.seed,
        }

    def num_views(self, scene_index: int) -> int:
        return self.views_per_scene

    def scene_parameters(self, scene_index: int) -> tuple[float, Tensor, Tensor]:
        """The ground truth of one scene.

        Returns:
            ``(radius, base_color, poses)``: the sphere's radius, its base
            colour of shape (3,), and the camera-to-world poses of all views
            with shape (num_views, 4, 4).
        """
        # Seeding from the scene index makes every call return the same
        # scene, no matter which views are requested or in which order.
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + scene_index)
        radius = torch.empty(1).uniform_(*_SPHERE_RADIUS_RANGE, generator=generator).item()
        base_color = torch.rand(3, generator=generator)
        poses = _sample_hemisphere_poses(self.views_per_scene, generator)
        return radius, base_color, poses

    def load_scene(self, scene_index: int, view_indices: Sequence[int] | None = None) -> SceneViews:
        radius, base_color, all_poses = self.scene_parameters(scene_index)

        if view_indices is None:
            view_indices = range(self.views_per_scene)
        poses = all_poses[list(view_indices)]

        focal = _FOCAL_LENGTH_IN_IMAGE_WIDTHS * self.image_size
        center = self.image_size / 2
        intrinsics = torch.tensor([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]])
        cameras = Cameras(camera_to_world=poses, intrinsics=intrinsics.expand(len(poses), 3, 3).clone())

        images = _render_sphere(cameras, self.image_size, radius, base_color)
        return SceneViews(
            images=images[None],
            cameras=Cameras(cameras.camera_to_world[None], cameras.intrinsics[None]),
            near=torch.tensor([_NEAR]),
            far=torch.tensor([_FAR]),
            scene_ids=[f"sphere_{scene_index:04d}"],
        )


def _sample_hemisphere_poses(num_views: int, generator: torch.Generator) -> Tensor:
    """Random cameras on the upper hemisphere (z up), all looking at the origin.

    Returns:
        Poses of shape (num_views, 4, 4).
    """
    azimuth = torch.rand(num_views, generator=generator) * 2 * math.pi
    # Stay away from the pole, where "up" in the image becomes ill-defined.
    elevation = torch.empty(num_views).uniform_(math.radians(10), math.radians(70), generator=generator)

    positions = _CAMERA_DISTANCE * torch.stack(
        [
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ],
        dim=-1,
    )
    origin = torch.zeros(3)
    world_up = torch.tensor([0.0, 0.0, 1.0])
    return torch.stack([look_at_pose(position, origin, world_up) for position in positions])


def _render_sphere(cameras: Cameras, image_size: int, radius: float, base_color: Tensor) -> Tensor:
    """Render a sphere centred at the origin by intersecting it with every pixel's ray.

    The surface colour blends the scene's base colour with a colour derived
    from the surface normal, so different sides of the sphere look different
    and the views are only consistent with one 3D arrangement.

    Args:
        cameras: Cameras with leading dimension (N,).
        image_size: Height and width of the output images.
        radius: Sphere radius.
        base_color: Shape (3,).

    Returns:
        Images of shape (N, 3, image_size, image_size).
    """
    num_views = cameras.camera_to_world.shape[0]
    pixels = pixel_centers(image_size, image_size)  # (H * W, 2)
    per_pixel_cameras = Cameras(
        camera_to_world=cameras.camera_to_world[:, None].expand(num_views, len(pixels), 4, 4),
        intrinsics=cameras.intrinsics[:, None].expand(num_views, len(pixels), 3, 3),
    )
    origins, directions = rays_through_pixels(per_pixel_cameras, pixels.expand(num_views, -1, -1))

    # Solve |origin + t * direction|^2 = radius^2 for the nearest t. With
    # unit-length directions this is t^2 + 2 b t + c = 0.
    b = (origins * directions).sum(dim=-1)
    c = (origins * origins).sum(dim=-1) - radius**2
    discriminant = b**2 - c
    hit = discriminant > 0
    distance = -b - torch.sqrt(discriminant.clamp(min=0))

    normals = (origins + distance[..., None] * directions) / radius
    surface_color = 0.6 * base_color + 0.4 * (0.5 + 0.5 * normals)
    white = torch.ones(3)
    colors = torch.where(hit[..., None], surface_color, white)

    return colors.reshape(num_views, image_size, image_size, 3).permute(0, 3, 1, 2)
