"""Input normalisation for image encoders.

The pipeline gives every model exactly the same images: RGB with values in
[0, 1]. That is what keeps the comparison fair. A pretrained encoder,
however, only works as intended on inputs normalised the way it was
pretrained, which differs between backbones. So each encoder normalises its
own input, as the first thing in its ``forward``, using this module:

    class MyEncoder(nn.Module):
        def __init__(self, input_normalization: str = "imagenet") -> None:
            super().__init__()
            self.normalize = InputNormalization(input_normalization)
            ...

        def forward(self, images):            # images in [0, 1]
            x = self.normalize(images)
            ...

Making it a constructor argument puts it in the model's config, where it is
visible and appears in the suite report's list of differences between runs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from scene_recon.registry import ConfigError

NORMALIZATIONS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # No change: the [0, 1] images as the pipeline provides them.
    "none": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    # The ImageNet statistics used by most pretrained backbones (torchvision
    # ResNets, ConvNeXt V2, Hiera). Check the source of your weights.
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # Maps [0, 1] to [-1, 1].
    "minus_one_to_one": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}
"""Name -> (per-channel mean, per-channel standard deviation). Add an entry to support new weights."""


class InputNormalization(nn.Module):
    """Computes ``(images - mean) / std`` per colour channel."""

    def __init__(self, name: str) -> None:
        """
        Args:
            name: One of the keys of ``NORMALIZATIONS``.
        """
        super().__init__()
        if name not in NORMALIZATIONS:
            raise ConfigError(f"Unknown input normalization '{name}'. Available: {sorted(NORMALIZATIONS)}.")
        self.name = name
        mean, std = NORMALIZATIONS[name]
        # Not saved in checkpoints (persistent=False): the config is the only
        # source of truth, so a checkpoint can never silently override it.
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1), persistent=False)

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: Shape (N, 3, H, W), values in [0, 1].
        """
        return (images - self.mean) / self.std

    def extra_repr(self) -> str:
        return self.name
