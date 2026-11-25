"""A pixelNeRF-style decoder: ResNet-block MLP with average pooling over views."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from scene_recon.registry import DECODERS


def positional_encoding(values: Tensor, num_frequencies: int) -> Tensor:
    """Append sines and cosines of the input at octave-spaced frequencies."""
    frequencies = math.pi * 2.0 ** torch.arange(num_frequencies, device=values.device)
    scaled = values[..., None] * frequencies  # (..., D, L)
    encodings = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-2)
    return torch.cat([values, encodings], dim=-1)


class ResnetBlock(nn.Module):
    """Two linear layers with a skip connection that adds the input back."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc0 = nn.Linear(width, width)
        self.fc1 = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.fc0(torch.relu(x))
        return x + self.fc1(torch.relu(hidden))


@DECODERS.register("pixelnerf_mlp")
class PixelNeRFDecoder(nn.Module):
    """Predicts colour and density for a 3D point from what each source view shows.

    Each view goes through a few ResNet blocks on its own. The views are then
    averaged, so any number of views works, and more blocks follow. This is a
    simplified version of the pixelNeRF decoder.

    Like pixelNeRF's, the average includes views that cannot see the point
    (behind the camera, or outside the image, where the feature at the
    border is read). This is kept on purpose, so that the encoder ablation
    uses pixelNeRF's decoder; it applies to every encoder alike.
    """

    def __init__(
        self,
        num_features: int,
        hidden_width: int = 512,
        num_frequencies: int = 6,
        num_blocks: int = 5,
        pool_after_block: int = 3,
    ) -> None:
        """
        Args:
            num_features: Size of the image feature vector for each point.
            hidden_width: Width of the hidden layers.
            num_frequencies: Number of sine and cosine frequencies that encode positions.
            num_blocks: Total number of ResNet blocks.
            pool_after_block: How many blocks run before averaging over views.
        """
        super().__init__()
        if not 0 <= pool_after_block <= num_blocks:
            raise ValueError("pool_after_block must be between 0 and num_blocks.")
        self.num_frequencies = num_frequencies

        position_size = 3 + 3 * 2 * num_frequencies
        direction_size = 3
        self.input_layer = nn.Linear(num_features + position_size + direction_size, hidden_width)
        blocks = [ResnetBlock(hidden_width) for _ in range(num_blocks)]
        self.blocks_per_view = nn.ModuleList(blocks[:pool_after_block])
        self.blocks_pooled = nn.ModuleList(blocks[pool_after_block:])
        self.output_layer = nn.Linear(hidden_width, 4)  # red, green, blue, density

    def forward(self, features: Tensor, points_camera: Tensor, directions_camera: Tensor):
        """
        All inputs: (B, V, P, ...).

        Args:
            features: Image features at each point, shape (B, V, P, num_features).
            points_camera: Points in each source camera's frame, shape (B, V, P, 3).
            directions_camera: Ray directions in each source camera's frame, shape (B, V, P, 3).

        Returns:
            ``(rgb, density)`` with shapes (B, P, 3) and (B, P).
        """
        encoded_points = positional_encoding(points_camera, self.num_frequencies)
        x = torch.cat([features, encoded_points, directions_camera], dim=-1)
        x = self.input_layer(x)

        for block in self.blocks_per_view:   # each view is processed on its own
            x = block(x)
        x = x.mean(dim=1)                     # average over views: (B, P, hidden)
        for block in self.blocks_pooled:     # the shared, combined part
            x = block(x)

        raw = self.output_layer(torch.relu(x))
        rgb = torch.sigmoid(raw[..., :3])
        density = torch.nn.functional.softplus(raw[..., 3])
        return rgb, density
