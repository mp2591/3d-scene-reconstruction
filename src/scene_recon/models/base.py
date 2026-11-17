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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

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
