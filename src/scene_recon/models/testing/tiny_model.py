"""A minimal pixel-aligned NeRF assembled from an encoder and a decoder. For testing only.

Besides exercising the pipeline, this file is a template for real models: it
shows one way for a top-level ``nn.Module`` to build its encoder and decoder
from config, so that swapping either one is a one-line config change::

    model:
      name: tiny_pixel_aligned_nerf
      encoder: {name: tiny_conv, num_features: 16}
      decoder: {name: tiny_pooled_mlp, hidden_width: 64}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor

from scene_recon.geometry import (
    project_to_pixels,
    rotate_directions_to_camera,
    sample_features_at_pixels,
    world_to_camera,
)
from scene_recon.models.base import ParameterGroup, ReconstructionModel
from scene_recon.registry import DECODERS, ENCODERS, MODELS
from scene_recon.structures import Cameras, FieldOutput, SceneViews


@dataclass
class _SceneEncoding:
    """What the model remembers about the source views between queries."""

    feature_maps: Tensor
    """Shape (B, V, C, feature_height, feature_width)."""

    cameras: Cameras
    """The source cameras, leading dimensions (B, V)."""

    image_size: tuple[int, int]
    """(height, width) of the source images, needed to look up features by pixel."""


@MODELS.register("tiny_pixel_aligned_nerf")
class TinyPixelAlignedNeRF(ReconstructionModel):
    """Conditions a small NeRF decoder on image features found by projection.

    To predict a 3D point, the point is projected into every source view, the
    encoder's feature map is read at that pixel, and the decoder combines the
    per-view features into colour and density. The same decoder is used for
    the coarse and the fine rendering pass.
    """

    def __init__(self, encoder: Mapping[str, Any], decoder: Mapping[str, Any]) -> None:
        """
        Args:
            encoder: Config section for the encoder, built from ``ENCODERS``.
            decoder: Config section for the decoder, built from ``DECODERS``.
                The decoder is told the encoder's feature size automatically.
        """
        super().__init__()
        self.encoder = ENCODERS.build(encoder)
        self.decoder = DECODERS.build(decoder, num_features=self.encoder.num_features)

    def encode_scene(self, source_views: SceneViews) -> _SceneEncoding:
        batch_size, num_views = source_views.num_scenes, source_views.num_views

        # The encoder sees each image independently, so scenes and views are
        # folded into one batch dimension and unfolded again afterwards.
        feature_maps = self.encoder(source_views.images.flatten(0, 1))
        feature_maps = feature_maps.unflatten(0, (batch_size, num_views))

        return _SceneEncoding(feature_maps, source_views.cameras, source_views.image_size)

    def query_field(
        self,
        scene_encoding: _SceneEncoding,
        points: Tensor,
        view_directions: Tensor,
        level: str,
    ) -> FieldOutput:
        batch_size, num_rays, num_samples, _ = points.shape
        source_poses = scene_encoding.cameras.camera_to_world

        # The decoder treats points independently, so the ray structure is
        # flattened to (B, P, 3) with P = R * S and restored at the end.
        flat_points = points.flatten(1, 2)
        flat_directions = view_directions[:, :, None].expand_as(points).flatten(1, 2)

        points_camera = world_to_camera(flat_points, source_poses)
        directions_camera = rotate_directions_to_camera(flat_directions, source_poses)
        pixels = project_to_pixels(points_camera, scene_encoding.cameras.intrinsics)
        features = sample_features_at_pixels(scene_encoding.feature_maps, pixels, scene_encoding.image_size)

        rgb, density = self.decoder(features, points_camera, directions_camera)
        return FieldOutput(
            rgb=rgb.reshape(batch_size, num_rays, num_samples, 3),
            density=density.reshape(batch_size, num_rays, num_samples),
        )

    def parameter_groups(self) -> list[ParameterGroup]:
        return [
            ParameterGroup(name="encoder", parameters=list(self.encoder.parameters())),
            ParameterGroup(name="decoder", parameters=list(self.decoder.parameters())),
        ]
