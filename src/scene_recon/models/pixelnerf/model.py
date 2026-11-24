"""A pixel-aligned NeRF built from an encoder and a decoder chosen in the config."""

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
from scene_recon.models.base import ParameterGroup, ReconstructionModel, build_encoder, encoder_parameter_groups
from scene_recon.registry import DECODERS, MODELS
from scene_recon.structures import Cameras, FieldOutput, SceneViews


@dataclass
class _SceneEncoding:
    feature_maps: Tensor  # (B, V, C, feature_height, feature_width)
    cameras: Cameras
    image_size: tuple[int, int]


@MODELS.register("pixel_aligned_nerf")
class PixelAlignedNeRF(ReconstructionModel):
    """A pixel aligned NeRF: image features found by projection feed a decoder.

    To predict a 3D point, the point is projected into every source view and the
    encoder's feature map is read at that pixel. The decoder turns the per view
    features into colour and density. The same decoder serves both rendering passes.

    The encoder must have the attributes that ``build_encoder`` checks:
    ``num_features``, the channels of its feature map; ``feature_offset``,
    where its feature grid sits in the image (see ``sample_features_at_pixels``);
    and ``adapter``, its layers that start from random values, or ``None``.

    Parameter groups, for separate learning rates: ``encoder`` (normally
    pretrained), ``adapter`` (if the encoder has one) and ``decoder``; see
    ``encoder_parameter_groups``.

    Args:
        encoder: Config section for the encoder, built from ``ENCODERS``.
        decoder: Config section for the decoder, built from ``DECODERS``. It is
            told the encoder's feature size automatically.
    """

    def __init__(self, encoder: Mapping[str, Any], decoder: Mapping[str, Any]) -> None:
        super().__init__()
        self.encoder = build_encoder(encoder, "pixel_aligned_nerf")
        self.decoder = DECODERS.build(decoder, num_features=self.encoder.num_features)

    def encode_scene(self, source_views: SceneViews) -> _SceneEncoding:
        batch_size, num_views = source_views.num_scenes, source_views.num_views
        feature_maps = self.encoder(source_views.images.flatten(0, 1))
        feature_maps = feature_maps.unflatten(0, (batch_size, num_views))
        return _SceneEncoding(feature_maps, source_views.cameras, source_views.image_size)

    def query_field(self, scene_encoding, points, view_directions, level) -> FieldOutput:
        batch_size, num_rays, num_samples, _ = points.shape
        source_poses = scene_encoding.cameras.camera_to_world

        flat_points = points.flatten(1, 2)
        flat_directions = view_directions[:, :, None].expand_as(points).flatten(1, 2)

        points_camera = world_to_camera(flat_points, source_poses)
        directions_camera = rotate_directions_to_camera(flat_directions, source_poses)
        pixels = project_to_pixels(points_camera, scene_encoding.cameras.intrinsics)
        features = sample_features_at_pixels(
            scene_encoding.feature_maps, pixels, scene_encoding.image_size, feature_offset=self.encoder.feature_offset
        )

        rgb, density = self.decoder(features, points_camera, directions_camera)
        return FieldOutput(
            rgb=rgb.reshape(batch_size, num_rays, num_samples, 3),
            density=density.reshape(batch_size, num_rays, num_samples),
        )

    def parameter_groups(self) -> list[ParameterGroup]:
        return [
            *encoder_parameter_groups(self.encoder),
            ParameterGroup(name="decoder", parameters=list(self.decoder.parameters())),
        ]
