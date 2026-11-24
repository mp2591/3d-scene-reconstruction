"""HieraNeRF: a Hiera-T encoder, the HieraNeRF bridge and a four-step attention decoder."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor

from scene_recon.models.base import ParameterGroup, ReconstructionModel, build_encoder, encoder_parameter_groups
from scene_recon.models.hiera.bridge import HieraNeRFBridge, HieraSceneEncoding
from scene_recon.registry import DECODERS, MODELS
from scene_recon.structures import FieldOutput, SceneViews


@MODELS.register("hiera_nerf")
class HieraNeRF(ReconstructionModel):
    """Reconstructs a scene from its source photos in one forward pass.

    ``encode_scene`` runs the encoder on every photo once. ``query_field``
    hands each 3D point to the bridge, which produces one embedding per
    photo, and the decoder turns those into density and colour, attending
    across photos and along each ray. One network serves both rendering
    passes (``level`` is not used).

    Parameter groups, for separate learning rates:

    * ``encoder``: the pretrained Hiera backbone (or the whole encoder, if its ``adapter`` is ``None``);
    * ``adapter``: the encoder's 1x1 convolution that fuses its stages, trained from scratch;
    * ``bridge`` and ``decoder``: trained from scratch.

    The first two come from ``encoder_parameter_groups``, as in ``pixel_aligned_nerf``.
    """

    def __init__(
        self,
        encoder: Mapping[str, Any],
        decoder: Mapping[str, Any],
        width: int = 128,
        position_frequencies: int = 6,
    ) -> None:
        """
        Args:
            encoder: Config section for the encoder, built from ``ENCODERS`` (normally ``hiera``).
                It must have ``num_features``, ``feature_offset`` and ``adapter``; see ``build_encoder``.
            decoder: Config section for the decoder, built from ``DECODERS``
                (normally ``hiera_ray_attention``). Its ``width`` is set from here.
            width: Channels of the tokens shared by the bridge and the decoder.
            position_frequencies: Sine/cosine frequencies encoding point positions in the bridge.
        """
        super().__init__()
        self.encoder = build_encoder(encoder, "hiera_nerf")
        self.bridge = HieraNeRFBridge(
            self.encoder.num_features, width, position_frequencies, feature_offset=self.encoder.feature_offset
        )
        self.decoder = DECODERS.build(decoder, width=width, geometry_size=self.bridge.geometry_size)

    def encode_scene(self, source_views: SceneViews) -> HieraSceneEncoding:
        batch_size, num_views = source_views.num_scenes, source_views.num_views
        feature_maps = self.encoder(source_views.images.flatten(0, 1)).unflatten(0, (batch_size, num_views))
        return HieraSceneEncoding(
            feature_maps=feature_maps,
            images=source_views.images,
            cameras=source_views.cameras,
            image_size=source_views.image_size,
            near=source_views.near,
            far=source_views.far,
        )

    def query_field(
        self, scene_encoding: HieraSceneEncoding, points: Tensor, view_directions: Tensor, level: str
    ) -> FieldOutput:
        _, num_rays, num_samples, _ = points.shape
        bridge = self.bridge(scene_encoding, points, view_directions)
        rgb, density = self.decoder(bridge, num_rays, num_samples)
        return FieldOutput(rgb=rgb, density=density)

    def parameter_groups(self) -> list[ParameterGroup]:
        return [
            *encoder_parameter_groups(self.encoder),
            ParameterGroup("bridge", list(self.bridge.parameters())),
            ParameterGroup("decoder", list(self.decoder.parameters())),
        ]
