"""The HieraNeRF bridge: from a 3D point to one embedding per source photo.

It starts as pixelNeRF does (project the point into each photo and read the
feature map there) with three changes:

1. **One world frame.** Positions and directions are expressed in the shared
   world frame rather than in each source camera's own frame, and the scene
   bounds (near and far) are part of the camera information: they set the
   scale of positions and normalise each point's depth in each photo.
2. **Feature, colour and direction per photo.** For each photo the bridge
   reads the feature and the colour at the point's projection and the
   direction from the photo's camera to the point, and combines them into
   one embedding.
3. **No averaging.** The embeddings of all photos are passed on as a list,
   one per photo, for the decoder to attend over.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scene_recon.geometry import project_to_pixels, sample_features_at_pixels, world_to_camera
from scene_recon.models.hiera.attention import encoding_size, sinusoidal_encoding
from scene_recon.structures import Cameras

GEOMETRY_SIZE = 8
"""Numbers describing how a photo sees a point: the direction from the photo's
camera to the point (3), its difference to the new viewing direction (3), their
cosine (1), and the point's normalised depth in the photo (1)."""


@dataclass
class HieraSceneEncoding:
    """What the model keeps about the source photos between queries."""

    feature_maps: Tensor
    """Shape (B, V, C, feature_height, feature_width)."""

    images: Tensor
    """The source photos, shape (B, V, 3, H, W), values in [0, 1]."""

    cameras: Cameras
    """Leading dimensions (B, V)."""

    image_size: tuple[int, int]
    """(height, width) of the photos."""

    near: Tensor
    """Shape (B,)."""

    far: Tensor
    """Shape (B,)."""


@dataclass
class BridgeOutput:
    """Everything the decoder gets for P = R * S points of each of B scenes.

    Shapes are (B, P, ...) for per-point values and (B, P, V, ...) for
    per-photo values.
    """

    point_tokens: Tensor
    """Embedding of each point's position, (B, P, width)."""

    view_tokens: Tensor
    """One embedding per photo combining feature, colour and geometry, (B, P, V, width)."""

    view_features: Tensor
    """The photo's image feature alone, projected to ``width``, (B, P, V, width)."""

    view_geometry: Tensor
    """How the photo sees the point (see ``GEOMETRY_SIZE``), (B, P, V, GEOMETRY_SIZE)."""

    view_colors: Tensor
    """The photo's colour at the point's projection, (B, P, V, 3)."""

    visible: Tensor
    """True where the point lies in front of the photo's camera and inside the photo, (B, P, V)."""

    ray_positions: Tensor
    """Position of each point along its ray, scaled by the scene bounds, (B, P, 1)."""

    target_directions: Tensor
    """The new viewing direction (world frame) at each point, (B, P, 3)."""


class HieraNeRFBridge(nn.Module):
    def __init__(self, num_features: int, width: int, position_frequencies: int, feature_offset: float) -> None:
        """
        Args:
            num_features: Channels of the encoder's feature maps.
            width: Channels of the tokens handed to the decoder.
            position_frequencies: Sine/cosine frequencies encoding point positions.
            feature_offset: Where the encoder's feature grid sits in the image
                (the encoder's ``feature_offset``; see ``sample_features_at_pixels``).
        """
        super().__init__()
        self.position_frequencies = position_frequencies
        self.feature_offset = feature_offset
        self.point_embedding = nn.Sequential(
            nn.Linear(encoding_size(3, position_frequencies), width), nn.GELU(), nn.Linear(width, width)
        )
        self.feature_projection = nn.Linear(num_features, width)
        self.view_embedding = nn.Sequential(
            nn.Linear(width + 3 + GEOMETRY_SIZE + 1, width), nn.GELU(), nn.Linear(width, width)
        )

    def forward(self, scene: HieraSceneEncoding, points: Tensor, view_directions: Tensor) -> BridgeOutput:
        """
        Args:
            scene: From ``HieraNeRF.encode_scene``.
            points: World positions, shape (B, R, S, 3).
            view_directions: Unit world directions of the rays, shape (B, R, 3).
        """
        batch_size, num_rays, num_samples, _ = points.shape
        flat_points = points.reshape(batch_size, num_rays * num_samples, 3)  # (B, P, 3)
        target_directions = (
            view_directions[:, :, None].expand(-1, -1, num_samples, -1).reshape(batch_size, -1, 3)
        )

        # Scene bounds: half the depth range sets the scale of positions
        # (about 0.5 for ShapeNet-SRN, so objects span roughly [-1, 1]).
        near, far = scene.near[:, None], scene.far[:, None]  # (B, 1)
        scale = ((far - near) / 2).clamp(min=1e-6)

        # Projection into every photo; the result's layout is (B, V, P, ...).
        poses = scene.cameras.camera_to_world
        points_camera = world_to_camera(flat_points, poses)
        pixels = project_to_pixels(points_camera, scene.cameras.intrinsics)
        # The feature grid may sit off the pixel grid; the colours and the
        # visibility test below use the photo's own pixels.
        features = sample_features_at_pixels(scene.feature_maps, pixels, scene.image_size, self.feature_offset)
        colors = sample_features_at_pixels(scene.images, pixels, scene.image_size)

        height, width = scene.image_size
        depth = points_camera[..., 2]
        visible = (
            (depth > 0)
            & (pixels[..., 0] >= 0) & (pixels[..., 0] <= width)
            & (pixels[..., 1] >= 0) & (pixels[..., 1] <= height)
        )

        # Directions from each photo's camera to the point, in the world frame.
        camera_centers = poses[..., :3, 3]  # (B, V, 3)
        to_point = F.normalize(flat_points[:, None] - camera_centers[:, :, None], dim=-1)  # (B, V, P, 3)
        target = target_directions[:, None]  # (B, 1, P, 3)
        normalized_depth = (depth - near[..., None]) / (far - near)[..., None].clamp(min=1e-6)
        geometry = torch.cat(
            [to_point, to_point - target, (to_point * target).sum(-1, keepdim=True), normalized_depth[..., None]],
            dim=-1,
        )

        def per_point(tensor: Tensor) -> Tensor:  # (B, V, P, ...) -> (B, P, V, ...)
            return tensor.transpose(1, 2)

        view_features = per_point(self.feature_projection(features))
        view_geometry = per_point(geometry)
        view_colors = per_point(colors)
        view_visible = per_point(visible)
        view_tokens = self.view_embedding(
            torch.cat([view_features, view_colors, view_geometry, view_visible[..., None].to(points.dtype)], dim=-1)
        )

        point_tokens = self.point_embedding(sinusoidal_encoding(flat_points / scale[..., None], self.position_frequencies))
        # Signed distance along the ray, measured from the point closest to the world origin.
        ray_positions = (flat_points * target_directions).sum(-1, keepdim=True) / scale[..., None]

        return BridgeOutput(
            point_tokens=point_tokens,
            view_tokens=view_tokens,
            view_features=view_features,
            view_geometry=view_geometry,
            view_colors=view_colors,
            visible=view_visible,
            ray_positions=ray_positions,
            target_directions=target_directions,
        )

    @property
    def geometry_size(self) -> int:
        return GEOMETRY_SIZE
