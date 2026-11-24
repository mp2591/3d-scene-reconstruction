"""The HieraNeRF decoder: attention across photos and along rays, in four steps."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scene_recon.models.hiera.attention import (
    AttentionBlock,
    HieraRayTransformer,
    encoding_size,
    sinusoidal_encoding,
)
from scene_recon.models.hiera.bridge import BridgeOutput
from scene_recon.registry import DECODERS


@DECODERS.register("hiera_ray_attention")
class HieraRayAttentionDecoder(nn.Module):
    """Predicts density and colour from the per-photo embeddings of the bridge.

    1. **Photos -> density representation.** The point's embedding and the
       photo embeddings attend to one another, which lets the network weigh
       which photos matter for this point. The point's token is the result.
    2. **Ray step (density).** The point is processed together with the other
       points of its ray by a ``HieraRayTransformer``. A linear layer gives
       the density.
    3. **Photos -> colour representation.** A query built from the density
       representation and the new viewing direction attends to the photos.
       Their features and directions form the keys, deciding which photos to
       trust; their colours form the values.
    4. **Ray step (colour).** A second ``HieraRayTransformer`` on the colour
       representations, then a linear layer gives red, green and blue.

    The position of each point along its ray is added with sines and cosines
    before both ray steps. Photos that do not see a point (it is behind the
    camera or projects outside the photo) are not attended to; if no photo
    sees it, all are.
    """

    def __init__(
        self,
        width: int,
        geometry_size: int,
        num_heads: int = 4,
        view_depth: int = 2,
        color_depth: int = 1,
        ray_window_size: int = 8,
        ray_local_depth: int = 1,
        ray_global_depth: int = 1,
        mlp_ratio: float = 2.0,
        ray_frequencies: int = 8,
        direction_frequencies: int = 4,
    ) -> None:
        """
        Args:
            width: Channels of every token; set by the top-level model.
            geometry_size: Numbers per photo describing its view of the point; set by the top-level model.
            num_heads: Attention heads in every block.
            view_depth: Blocks of step 1 (point and photos attend to one another).
            color_depth: Blocks of step 3 (colour query attends to the photos).
            ray_window_size: Samples per local group in the ray steps.
            ray_local_depth: Local blocks in each ray step.
            ray_global_depth: Global blocks in each ray step.
            mlp_ratio: Hidden width of every block's MLP, relative to ``width``.
            ray_frequencies: Sine/cosine frequencies encoding the position along the ray.
            direction_frequencies: Sine/cosine frequencies encoding the new viewing direction.
        """
        super().__init__()
        self.ray_frequencies = ray_frequencies
        self.direction_frequencies = direction_frequencies

        # Step 1.
        self.view_blocks = nn.ModuleList(
            AttentionBlock(width, num_heads, mlp_ratio) for _ in range(view_depth)
        )

        # Step 2.
        ray_position_size = encoding_size(1, ray_frequencies)
        self.density_ray_position = nn.Linear(ray_position_size, width)
        self.density_ray = HieraRayTransformer(
            width, num_heads, ray_window_size, ray_local_depth, ray_global_depth, mlp_ratio
        )
        self.density_norm = nn.LayerNorm(width)
        self.density_head = nn.Linear(width, 1)

        # Step 3.
        self.color_query = nn.Linear(width + encoding_size(3, direction_frequencies), width)
        self.color_key = nn.Linear(width + geometry_size, width)
        self.color_value = nn.Linear(3, width)
        self.color_blocks = nn.ModuleList(
            AttentionBlock(width, num_heads, mlp_ratio, cross=True) for _ in range(color_depth)
        )

        # Step 4.
        self.color_ray_position = nn.Linear(ray_position_size, width)
        self.color_ray = HieraRayTransformer(
            width, num_heads, ray_window_size, ray_local_depth, ray_global_depth, mlp_ratio
        )
        self.color_norm = nn.LayerNorm(width)
        self.rgb_head = nn.Linear(width, 3)

    def forward(self, bridge: BridgeOutput, num_rays: int, num_samples: int) -> tuple[Tensor, Tensor]:
        """
        Args:
            bridge: The bridge's output for P = R * S points per scene.
            num_rays: R.
            num_samples: S, samples per ray, ordered near to far.

        Returns:
            ``(rgb, density)`` with shapes (B, R, S, 3) and (B, R, S).
        """
        batch_size, num_points, num_views, width = bridge.view_tokens.shape
        flat = batch_size * num_points

        visible = bridge.visible.reshape(flat, num_views)
        attend_to = visible | ~visible.any(dim=-1, keepdim=True)
        ray_position = sinusoidal_encoding(bridge.ray_positions, self.ray_frequencies)
        ray_position = ray_position.reshape(batch_size * num_rays, num_samples, -1)

        # Step 1: the point and its photos attend to one another.
        tokens = torch.cat(
            [bridge.point_tokens.reshape(flat, 1, width), bridge.view_tokens.reshape(flat, num_views, width)], dim=1
        )
        mask = torch.cat([attend_to.new_ones(flat, 1), attend_to], dim=1)
        for block in self.view_blocks:
            tokens = block(tokens, mask=mask)
        density_repr = tokens[:, 0].reshape(batch_size * num_rays, num_samples, width)

        # Step 2: along the ray, then density.
        density_repr = self.density_ray(density_repr + self.density_ray_position(ray_position))
        density = F.softplus(self.density_head(self.density_norm(density_repr))).squeeze(-1)

        # Step 3: a colour query attends to the photos.
        direction = sinusoidal_encoding(bridge.target_directions, self.direction_frequencies)
        query = self.color_query(torch.cat([density_repr.reshape(flat, width), direction.reshape(flat, -1)], dim=-1))
        keys = self.color_key(torch.cat([bridge.view_features, bridge.view_geometry], dim=-1))
        values = self.color_value(bridge.view_colors)
        query = query[:, None]
        keys, values = keys.reshape(flat, num_views, width), values.reshape(flat, num_views, width)
        for block in self.color_blocks:
            query = block(query, keys=keys, values=values, mask=attend_to)
        color_repr = query.reshape(batch_size * num_rays, num_samples, width)

        # Step 4: along the ray, then colour.
        color_repr = self.color_ray(color_repr + self.color_ray_position(ray_position))
        rgb = torch.sigmoid(self.rgb_head(self.color_norm(color_repr)))

        return (
            rgb.reshape(batch_size, num_rays, num_samples, 3),
            density.reshape(batch_size, num_rays, num_samples),
        )
