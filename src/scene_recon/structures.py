"""The data containers that flow between pipeline stages.

Shape letters used throughout the code base:

    B  scenes in a batch
    N  views of a scene (V when they are the source views given to the model)
    R  rays per scene
    S  samples along a ray
    H, W  image height and width in pixels

Camera convention (OpenCV): in the camera frame +x points right, +y points
down and +z points forward into the scene. Poses are camera-to-world
matrices. This is the convention of the ShapeNet-SRN pose files, so they are
used exactly as stored on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass
class Cameras:
    """Pinhole cameras. Both tensors share the same leading dimensions."""

    camera_to_world: Tensor
    """Poses of shape (..., 4, 4) mapping camera coordinates to world coordinates."""

    intrinsics: Tensor
    """Matrices of shape (..., 3, 3), ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``, in pixels."""

    def to(self, device: torch.device | str) -> Cameras:
        return Cameras(self.camera_to_world.to(device), self.intrinsics.to(device))


@dataclass
class SceneViews:
    """A batch of scenes, each seen from the same number of posed views.

    The same container is used for everything a dataset loads and for the
    subset of views handed to the model as its sparse input.
    """

    images: Tensor
    """RGB images of shape (B, N, 3, H, W) with values in [0, 1]."""

    cameras: Cameras
    """One camera per image; leading dimensions (B, N)."""

    near: Tensor
    """Shape (B,). Distance along a ray at which the scene content starts."""

    far: Tensor
    """Shape (B,). Distance along a ray at which the scene content ends."""

    scene_ids: list[str]
    """One human-readable identifier per scene."""

    @property
    def num_scenes(self) -> int:
        return self.images.shape[0]

    @property
    def num_views(self) -> int:
        return self.images.shape[1]

    @property
    def image_size(self) -> tuple[int, int]:
        """(height, width) in pixels."""
        return self.images.shape[-2], self.images.shape[-1]

    def to(self, device: torch.device | str) -> SceneViews:
        return SceneViews(
            images=self.images.to(device),
            cameras=self.cameras.to(device),
            near=self.near.to(device),
            far=self.far.to(device),
            scene_ids=self.scene_ids,
        )

    def select_views(self, view_indices: Sequence[int] | Tensor) -> SceneViews:
        """Keep only the given views (the same ones for every scene in the batch)."""
        indices = torch.as_tensor(view_indices, dtype=torch.long, device=self.images.device)
        return SceneViews(
            images=self.images[:, indices],
            cameras=Cameras(
                camera_to_world=self.cameras.camera_to_world[:, indices],
                intrinsics=self.cameras.intrinsics[:, indices],
            ),
            near=self.near,
            far=self.far,
            scene_ids=self.scene_ids,
        )

    @staticmethod
    def concatenate(scenes: Sequence[SceneViews]) -> SceneViews:
        """Join several batches into one. All scenes must have the same number of views."""
        return SceneViews(
            images=torch.cat([scene.images for scene in scenes]),
            cameras=Cameras(
                camera_to_world=torch.cat([scene.cameras.camera_to_world for scene in scenes]),
                intrinsics=torch.cat([scene.cameras.intrinsics for scene in scenes]),
            ),
            near=torch.cat([scene.near for scene in scenes]),
            far=torch.cat([scene.far for scene in scenes]),
            scene_ids=[scene_id for scene in scenes for scene_id in scene.scene_ids],
        )


@dataclass
class RayBundle:
    """Rays in world coordinates, R per scene."""

    origins: Tensor
    """Shape (B, R, 3)."""

    directions: Tensor
    """Shape (B, R, 3), unit length. Depths are therefore distances along the ray."""

    near: Tensor
    """Shape (B,)."""

    far: Tensor
    """Shape (B,)."""

    @property
    def num_rays(self) -> int:
        return self.origins.shape[1]

    def slice(self, start: int, stop: int) -> RayBundle:
        """Rays ``start`` to ``stop`` of every scene, for rendering in chunks."""
        return RayBundle(
            origins=self.origins[:, start:stop],
            directions=self.directions[:, start:stop],
            near=self.near,
            far=self.far,
        )


@dataclass
class FieldOutput:
    """What a model predicts at a set of 3D sample points."""

    rgb: Tensor
    """Shape (B, R, S, 3) with values in [0, 1]."""

    density: Tensor
    """Shape (B, R, S), non-negative volume density (sigma)."""


@dataclass
class RenderOutput:
    """The result of volume rendering a set of rays."""

    rgb: Tensor
    """Shape (B, R, 3). Rendered colour, composited over the background colour."""

    depth: Tensor
    """Shape (B, R). Expected distance along the ray."""

    opacity: Tensor
    """Shape (B, R). Total weight along the ray: 0 for empty space, 1 for a solid hit."""

    weights: Tensor
    """Shape (B, R, S). Contribution of each sample to its ray's colour."""

    sample_depths: Tensor
    """Shape (B, R, S). Distance along the ray of each sample, in increasing order."""
