"""A small MLP decoder that averages information across views. For testing only."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from scene_recon.registry import DECODERS


@DECODERS.register("tiny_pooled_mlp")
class TinyPooledMLPDecoder(nn.Module):
    """Predicts colour and density for a point from its per-view observations.

    Follows the outline of the pixelNeRF decoder at a fraction of the size:
    one MLP processes each source view's observation independently, the
    results are averaged over views, and a second MLP turns the average into
    colour and density. Averaging is what lets the same weights handle any
    number of source views.
    """

    def __init__(
        self, num_features: int, hidden_width: int = 64, num_frequencies: int = 4, dropout: float = 0.0
    ) -> None:
        """
        Args:
            num_features: Size of the image feature vector sampled for each point.
            hidden_width: Width of the hidden layers.
            num_frequencies: Number of sine/cosine frequencies used to encode point positions.
            dropout: Dropout probability after the per-view layers. Exists so that
                tests can check training with randomness inside the model.
        """
        super().__init__()
        self.num_frequencies = num_frequencies

        position_size = 3 + 3 * 2 * num_frequencies
        direction_size = 3
        self.per_view_mlp = nn.Sequential(
            nn.Linear(num_features + position_size + direction_size, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pooled_mlp = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 4),  # red, green, blue, density
        )

    def forward(self, features: Tensor, points_camera: Tensor, directions_camera: Tensor) -> tuple[Tensor, Tensor]:
        """
        All inputs describe the same P points as observed from each of V source views.

        Args:
            features: Image features sampled at each point's projection, shape (B, V, P, num_features).
            points_camera: Point positions in each source camera's frame, shape (B, V, P, 3).
            directions_camera: Viewing direction of each point's ray in each
                source camera's frame, shape (B, V, P, 3).

        Returns:
            ``(rgb, density)`` with shapes (B, P, 3) and (B, P). RGB is in
            [0, 1] and density is non-negative.
        """
        encoded_points = positional_encoding(points_camera, self.num_frequencies)
        per_view_input = torch.cat([features, encoded_points, directions_camera], dim=-1)

        per_view_hidden = self.per_view_mlp(per_view_input)  # (B, V, P, hidden)
        pooled = per_view_hidden.mean(dim=1)  # (B, P, hidden)
        raw_output = self.pooled_mlp(pooled)

        rgb = torch.sigmoid(raw_output[..., :3])
        density = torch.relu(raw_output[..., 3])
        return rgb, density


def positional_encoding(values: Tensor, num_frequencies: int) -> Tensor:
    """Append sines and cosines of the input at octave-spaced frequencies (NeRF, Sec. 5.1).

    An MLP struggles to represent fine detail as a function of raw
    coordinates; this encoding makes high-frequency variation easy to express.

    Args:
        values: Shape (..., D).
        num_frequencies: Number of frequencies L.

    Returns:
        Shape (..., D + D * 2 * L): the input followed by its encodings.
    """
    frequencies = math.pi * 2.0 ** torch.arange(num_frequencies, device=values.device)
    scaled = values[..., None] * frequencies  # (..., D, L)
    encodings = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-2)
    return torch.cat([values, encodings], dim=-1)
