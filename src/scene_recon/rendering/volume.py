"""The volume rendering integral, in its discrete form (NeRF, Mildenhall et al., 2020)."""

from __future__ import annotations

import torch
from torch import Tensor

from scene_recon.structures import FieldOutput, RenderOutput


def composite_along_rays(
    field: FieldOutput, sample_depths: Tensor, far: Tensor, background_color: Tensor
) -> RenderOutput:
    """Turn per-sample colours and densities into one colour per ray.

    Each sample stands for the ray segment up to the next sample, and the
    last one for the segment up to ``far``, as in pixelNeRF (the original
    NeRF makes the last segment infinite). A segment
    of length ``delta`` and density ``sigma`` absorbs a fraction
    ``alpha = 1 - exp(-sigma * delta)`` of the light that reaches it. A
    sample's weight is its ``alpha`` times the fraction of light that
    survived all the segments in front of it (the transmittance).

    Args:
        field: Model predictions, ``rgb`` (B, R, S, 3) and ``density`` (B, R, S).
        sample_depths: Shape (B, R, S), increasing along the last dimension.
        far: Far distance of each scene's rays, shape (B,).
        background_color: Shape (3,). Shown through whatever the ray did not absorb.
    """
    segment_lengths = sample_depths[..., 1:] - sample_depths[..., :-1]
    last_segment = far[:, None, None] - sample_depths[..., -1:]
    segment_lengths = torch.cat([segment_lengths, last_segment], dim=-1)

    alpha = 1.0 - torch.exp(-field.density * segment_lengths)

    # Transmittance in front of sample i is the product of (1 - alpha) over
    # samples 0 .. i-1, so the product is shifted by one: nothing is in front
    # of the first sample.
    light_passed = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    transmittance = torch.cat([torch.ones_like(light_passed[..., :1]), light_passed[..., :-1]], dim=-1)
    weights = alpha * transmittance

    opacity = weights.sum(dim=-1)
    rgb = (weights[..., None] * field.rgb).sum(dim=-2)
    rgb = rgb + (1.0 - opacity[..., None]) * background_color
    depth = (weights * sample_depths).sum(dim=-1)

    return RenderOutput(rgb=rgb, depth=depth, opacity=opacity, weights=weights, sample_depths=sample_depths)
