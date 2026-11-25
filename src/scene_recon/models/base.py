"""The contract between the pipeline and a neural network.

The pipeline treats the network as a black box with two operations:

1. ``encode_scene`` looks at the sparse input views once per scene.
2. ``query_field`` answers "what colour and density is at these 3D points?"
   and is called many times per scene, once per chunk of rays.

Everything else (ray casting, sampling points along rays, volume rendering,
the loss, optimisation, evaluation) is done by the pipeline. How a model
splits its work between an encoder and a decoder, or whether it makes that
split at all, is up to the model. See ``docs/writing_a_model.md``.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor, nn

from scene_recon.registry import ENCODERS, ConfigError
from scene_recon.structures import FieldOutput, SceneViews

# Names of the two rendering passes. The coarse pass samples points evenly
# along each ray; the fine pass adds samples where the coarse pass found
# content. A model may use separate networks for the two or share one.
LEVEL_COARSE = "coarse"
LEVEL_FINE = "fine"


@dataclass
class ParameterGroup:
    """A named set of parameters that is optimised with its own learning rate."""

    name: str
    """Used to address the group from the config, e.g. ``"encoder"``.

    Dots create a hierarchy: settings given for ``"encoder"`` in the config
    also apply to groups named ``"encoder.stage0"``, ``"encoder.stage1"``, ...
    """

    parameters: list[nn.Parameter]

    lr_scale: float = 1.0
    """Multiplier on the group's configured learning rate.

    This is the hook for layer-wise learning-rate decay: give each encoder
    stage its own group with a progressively smaller ``lr_scale``.
    """


class ReconstructionModel(nn.Module, ABC):
    """Base class for every top-level model the pipeline can train and evaluate."""

    @abstractmethod
    def encode_scene(self, source_views: SceneViews) -> Any:
        """Process the input views of each scene, once.

        Args:
            source_views: B scenes with V source views each. V varies
                between calls (the proposal uses 1 to 3), so the model must
                not assume a fixed number of views.

        Returns:
            Anything the model wants to remember about the scenes, such as
            feature maps together with the source cameras. The pipeline
            passes it back to ``query_field`` untouched.
        """

    @abstractmethod
    def query_field(
        self,
        scene_encoding: Any,
        points: Tensor,
        view_directions: Tensor,
        level: str,
    ) -> FieldOutput:
        """Predict colour and density at 3D points.

        The points keep their arrangement along rays so that models can
        reason across the samples of a ray (as the TransNeRF ray decoder
        does). Along the S dimension, points are ordered from near to far.

        Args:
            scene_encoding: The value returned by ``encode_scene``.
            points: World-space positions of shape (B, R, S, 3).
            view_directions: Unit-length world-space direction of each ray,
                shape (B, R, 3).
            level: ``LEVEL_COARSE`` or ``LEVEL_FINE``.

        Returns:
            ``FieldOutput`` with ``rgb`` of shape (B, R, S, 3) in [0, 1] and
            non-negative ``density`` of shape (B, R, S).
        """

    def parameter_groups(self) -> list[ParameterGroup]:
        """Split the parameters into groups that get separate optimiser settings.

        The default puts everything in one group called ``"model"``.
        Override it to, for example, train a pretrained encoder with a lower
        learning rate and warmup. Every trainable parameter must appear in
        exactly one group.
        """
        return [ParameterGroup(name="model", parameters=list(self.parameters()))]


ENCODER_ATTRIBUTES = ("num_features", "feature_offset", "adapter", "pretrained_weights")
"""What a pixel-aligned model reads from its image encoder:

* ``num_features``: the channels of its feature map;
* ``feature_offset``: where its feature grid sits in the image (see
  ``sample_features_at_pixels``);
* ``adapter``: the module holding the layers the encoder adds on top of
  pretrained weights, which start from random values (such as a 1x1
  convolution that fuses several stages), or ``None`` if it adds none.
  ``encoder_parameter_groups`` trains them in a group of their own;
* ``pretrained_weights``: the weights it starts from, as
  ``describe_pretrained_weights`` gives them, or ``None`` if it starts from
  random values. Checkpoints record them (see ``pretrained_weights_of``).
"""


def build_encoder(section: Mapping[str, Any], model_name: str) -> nn.Module:
    """Build an image encoder from its config section and check it has ``ENCODER_ATTRIBUTES``.

    Args:
        section: The encoder's config section, built from ``ENCODERS``.
        model_name: The registered name of the model building it, for the error message.

    Raises:
        ConfigError: If the encoder lacks one of the attributes.
    """
    encoder = ENCODERS.build(section)
    for attribute in ENCODER_ATTRIBUTES:
        if not hasattr(encoder, attribute):
            raise ConfigError(
                f"The encoder '{section['name']}' has no attribute '{attribute}', which {model_name} "
                "needs (see docs/writing_a_model.md)."
            )
    return encoder


def encoder_parameter_groups(encoder: nn.Module) -> list[ParameterGroup]:
    """The parameter groups of an encoder made by ``build_encoder``.

    * ``encoder``: every parameter outside the adapter, normally pretrained;
    * ``adapter``: the encoder's ``adapter``, if it has one.

    The adapter starts from random values, so it is not part of ``encoder``:
    a config that gives the pretrained ``encoder`` group a lower learning
    rate and a warmup leaves the adapter at the base learning rate, like the
    rest of the new layers.
    """
    if encoder.adapter is None:
        return [ParameterGroup(name="encoder", parameters=list(encoder.parameters()))]
    adapter_ids = {id(parameter) for parameter in encoder.adapter.parameters()}
    return [
        ParameterGroup(name="encoder", parameters=[p for p in encoder.parameters() if id(p) not in adapter_ids]),
        ParameterGroup(name="adapter", parameters=list(encoder.adapter.parameters())),
    ]


def describe_pretrained_weights(source: str, module: nn.Module) -> dict[str, str]:
    """What an encoder records as its ``pretrained_weights``.

    Call it right after loading the weights, so the hash is of the weights as
    loaded, before any training.

    Args:
        source: Where the weights came from, as the config names them: a file
            path, or a ``timm`` model name.
        module: The part that holds the loaded weights, without the
            ``adapter`` (whose random start differs from seed to seed).

    Returns:
        ``{"source": source, "sha256": ...}``. Two runs that start from the
        same weights have the same hash, however the file was obtained.
    """
    return {"source": source, "sha256": weights_sha256(module)}


def pretrained_weights_of(model: nn.Module) -> dict[str, dict[str, str]]:
    """The pretrained weights a model starts from, by the name of the part that holds them.

    Collected from every part with a ``pretrained_weights`` attribute that is
    not ``None`` (every encoder made by ``build_encoder`` has the attribute),
    e.g. ``{"encoder": {"source": "weights/hiera_tiny_224_mae_in1k.pth",
    "sha256": "..."}}``. Empty if the whole model starts from random values.
    """
    return {
        name: module.pretrained_weights
        for name, module in model.named_modules()
        if getattr(module, "pretrained_weights", None) is not None
    }


def weights_sha256(module: nn.Module) -> str:
    """SHA-256 of a module's saved state (parameters and persistent buffers): which weights it holds."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        values = tensor.detach().cpu().contiguous()
        digest.update(f"{name}\t{values.dtype}\t{tuple(values.shape)}\n".encode())
        digest.update(values.numpy().tobytes() if values.numel() else b"")
    return digest.hexdigest()
