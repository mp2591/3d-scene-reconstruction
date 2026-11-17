"""Choosing where along each ray to query the model.

Random draws come from an explicit CPU ``torch.Generator`` and are then moved
to the device, never from PyTorch's global generator, which belongs to the
model (see ``scene_recon.data.schedule``). Passing no generator gives
deterministic samples, as used for evaluation.
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
            from this generator (used in training, so that the model sees a
            continuum of depths over time). If ``None``, take the bin centre
            (used in evaluation, to keep renders deterministic).

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
    depths: Tensor, weights: Tensor, num_samples: int, generator: torch.Generator | None
) -> Tensor:
    """Draw extra samples where a previous rendering pass found content.

    This is the hierarchical sampling of the original NeRF. The rendering
    weights of the coarse pass are treated as a histogram over depth, whose
    bin edges lie midway between neighbouring coarse samples, and new depths
    are drawn from it by inverse-transform sampling.

    Args:
        depths: Depths of the previous pass, shape (B, R, S) with S >= 3.
        weights: Rendering weights of the previous pass, shape (B, R, S).
        num_samples: Number of new samples per ray.
        generator: If given, draw randomly from it. If ``None``, use evenly
            spaced quantiles of the histogram so the result is deterministic.

    Returns:
        New depths of shape (B, R, num_samples), increasing along the last
        dimension. No gradient flows back into ``weights``.
    """
    weights = weights.detach()

    # The first and last sample have no bin of their own between midpoints,
    # so their weights are dropped: S samples give S - 1 edges and S - 2 bins.
    bin_edges = 0.5 * (depths[..., 1:] + depths[..., :-1])
    bin_weights = weights[..., 1:-1] + 1e-5  # the epsilon keeps empty rays well-defined

    probabilities = bin_weights / bin_weights.sum(dim=-1, keepdim=True)
    cumulative = torch.cumsum(probabilities, dim=-1)
    cumulative = torch.cat([torch.zeros_like(cumulative[..., :1]), cumulative], dim=-1)  # (B, R, S - 1)

    if generator is not None:
        quantiles = torch.rand(*depths.shape[:-1], num_samples, generator=generator).to(depths.device)
        quantiles, _ = torch.sort(quantiles, dim=-1)
    else:
        quantiles = torch.linspace(0.0, 1.0, num_samples + 2, device=depths.device)[1:-1]
        quantiles = quantiles.expand(*depths.shape[:-1], num_samples).contiguous()

    # For each quantile, find the bin whose cumulative range contains it...
    num_bins = bin_weights.shape[-1]
    bin_index = torch.searchsorted(cumulative, quantiles, right=True) - 1
    bin_index = bin_index.clamp(0, num_bins - 1)

    # ...and place the sample inside the bin by linear interpolation.
    cumulative_start = torch.gather(cumulative, -1, bin_index)
    cumulative_end = torch.gather(cumulative, -1, bin_index + 1)
    edge_start = torch.gather(bin_edges, -1, bin_index)
    edge_end = torch.gather(bin_edges, -1, bin_index + 1)

    position_in_bin = (quantiles - cumulative_start) / (cumulative_end - cumulative_start)
    position_in_bin = position_in_bin.clamp(0.0, 1.0)  # guards against rounding in the cumulative sum
    return edge_start + position_in_bin * (edge_end - edge_start)
