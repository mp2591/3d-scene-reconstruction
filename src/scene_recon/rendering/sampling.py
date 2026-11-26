"""Choosing where along each ray to query the model.

Random draws come from an explicit CPU ``torch.Generator`` and are then moved
to the device, never from PyTorch's global generator, which belongs to the
model (see ``scene_recon.data.schedule``). Training and evaluation both pass
one, as pixelNeRF draws its samples at random in both. Passing none gives
samples at fixed positions.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scene_recon.structures import RayBundle


def sample_stratified_depths(rays: RayBundle, num_samples: int, generator: torch.Generator | None) -> Tensor:
    """Spread samples evenly between each ray's near and far distance.

    The range is divided into ``num_samples`` equal bins and one depth is
    taken from each.

    Args:
        rays: The rays to sample along.
        num_samples: Samples per ray, S.
        generator: If given, take a random depth within each bin, drawn
            from this generator, so that the model sees a continuum of
            depths over time. If ``None``, take the bin centre.

    Returns:
        Depths of shape (B, R, S), increasing along the last dimension.
    """
    batch_size, num_rays = rays.origins.shape[:2]
    device = rays.origins.device

    if generator is not None:
        position_in_bin = torch.rand(batch_size, num_rays, num_samples, generator=generator).to(device)
    else:
        position_in_bin = torch.full((batch_size, num_rays, num_samples), 0.5, device=device)

    bin_index = torch.arange(num_samples, device=device)
    fraction = (bin_index + position_in_bin) / num_samples  # in [0, 1]

    near = rays.near[:, None, None]
    far = rays.far[:, None, None]
    return near + (far - near) * fraction


def sample_importance_depths(
    rays: RayBundle, weights: Tensor, num_samples: int, generator: torch.Generator | None
) -> Tensor:
    """Draw extra samples where the coarse pass found content, as pixelNeRF does.

    The range between near and far is divided into the same S equal bins as
    the coarse samples (see ``sample_stratified_depths``), and each bin gets
    the rendering weight of its coarse sample. A new depth picks a bin with
    a probability in proportion to that weight, and lies anywhere inside it.

    Args:
        rays: The rays the coarse pass rendered.
        weights: Rendering weights of the coarse pass, shape (B, R, S).
        num_samples: Number of new samples per ray.
        generator: If given, draw randomly from it. If ``None``, use evenly
            spaced quantiles.

    Returns:
        New depths of shape (B, R, num_samples), increasing along the last
        dimension. No gradient flows back into ``weights``.
    """
    num_bins = weights.shape[-1]
    bin_weights = weights.detach() + 1e-5  # the epsilon keeps rays that hit nothing well-defined

    probabilities = bin_weights / bin_weights.sum(dim=-1, keepdim=True)
    cumulative = torch.cumsum(probabilities, dim=-1)
    cumulative = torch.cat([torch.zeros_like(cumulative[..., :1]), cumulative], dim=-1)  # (B, R, S + 1)

    if generator is not None:
        quantiles = torch.rand(*weights.shape[:-1], num_samples, generator=generator).to(weights.device)
        quantiles, _ = torch.sort(quantiles, dim=-1)
    else:
        quantiles = torch.linspace(0.0, 1.0, num_samples + 2, device=weights.device)[1:-1]
        quantiles = quantiles.expand(*weights.shape[:-1], num_samples).contiguous()

    # For each quantile, find the bin whose cumulative range contains it...
    bin_index = torch.searchsorted(cumulative, quantiles, right=True) - 1
    bin_index = bin_index.clamp(0, num_bins - 1)

    # ...and place the sample as far into the bin as the quantile is into the
    # bin's cumulative range. For a random quantile that is anywhere in the
    # bin, with equal chance, as in pixelNeRF.
    cumulative_start = torch.gather(cumulative, -1, bin_index)
    cumulative_end = torch.gather(cumulative, -1, bin_index + 1)
    position_in_bin = (quantiles - cumulative_start) / (cumulative_end - cumulative_start)
    position_in_bin = position_in_bin.clamp(0.0, 1.0)  # guards against rounding in the cumulative sum
    fraction = (bin_index + position_in_bin) / num_bins  # in [0, 1]

    near = rays.near[:, None, None]
    far = rays.far[:, None, None]
    return near + (far - near) * fraction


def sample_around_depth(
    rays: RayBundle, depth: Tensor, num_samples: int, spread: float, generator: torch.Generator | None
) -> Tensor:
    """Place samples close to the depth the coarse pass found, as pixelNeRF does.

    The samples are spread normally around ``depth`` and kept between near
    and far.

    Args:
        rays: The rays the coarse pass rendered.
        depth: Expected depth of each ray in the coarse pass, shape (B, R).
            Gradients flow back through it, as in pixelNeRF.
        num_samples: Samples per ray.
        spread: Standard deviation of the samples around ``depth``.
        generator: If given, draw the offsets randomly from it. If ``None``,
            use evenly spaced quantiles of the normal distribution.

    Returns:
        Depths of shape (B, R, num_samples), increasing along the last dimension.
    """
    if generator is not None:
        offsets = torch.randn(*depth.shape, num_samples, generator=generator).to(depth.device)
        offsets, _ = torch.sort(offsets, dim=-1)
    else:
        quantiles = (torch.arange(num_samples, device=depth.device) + 0.5) / num_samples
        offsets = torch.special.ndtri(quantiles).expand(*depth.shape, num_samples)

    depths = depth[..., None] + spread * offsets
    near = rays.near[:, None, None]
    far = rays.far[:, None, None]
    return torch.maximum(torch.minimum(depths, far), near)
