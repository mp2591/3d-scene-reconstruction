"""Attention building blocks of the HieraNeRF decoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AttentionBlock(nn.Module):
    """A pre-norm transformer block: multi-head attention, then an MLP, each with a residual.

    Used for self-attention (``keys`` and ``values`` omitted) or for
    cross-attention, where keys and values may come from different inputs.

    Attention is written out instead of calling
    ``F.scaled_dot_product_attention``: the decoder attends over very many
    short sequences (one per 3D point), a shape for which some fused GPU
    kernels fail once the batch exceeds 65,535 sequences.
    """

    def __init__(self, width: int, num_heads: int, mlp_ratio: float = 2.0, cross: bool = False) -> None:
        """
        Args:
            width: Channels of every token.
            num_heads: Attention heads; must divide ``width``.
            mlp_ratio: Hidden width of the MLP, relative to ``width``.
            cross: Cross-attention, with separate key and value inputs.
        """
        super().__init__()
        if width % num_heads:
            raise ValueError(f"width ({width}) must be a multiple of num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.cross = cross
        self.norm_query = nn.LayerNorm(width)
        self.to_query = nn.Linear(width, width)
        if cross:
            self.norm_key = nn.LayerNorm(width)
            self.norm_value = nn.LayerNorm(width)
            self.to_key = nn.Linear(width, width)
            self.to_value = nn.Linear(width, width)
        else:
            self.to_key_value = nn.Linear(width, 2 * width)
        self.to_output = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(
        self, x: Tensor, keys: Tensor | None = None, values: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        """
        Args:
            x: Query tokens, shape (N, Lq, width).
            keys: Key tokens, shape (N, Lk, width). Cross-attention only.
            values: Value tokens, shape (N, Lk, width). Cross-attention only.
            mask: Shape (N, Lk), True where a key may be attended to. Every
                row needs at least one True.

        Returns:
            Shape (N, Lq, width).
        """
        normed = self.norm_query(x)
        query = self.to_query(normed)
        if self.cross:
            if keys is None or values is None:
                raise ValueError("A cross-attention block needs keys and values.")
            key, value = self.to_key(self.norm_key(keys)), self.to_value(self.norm_value(values))
        else:
            key, value = self.to_key_value(normed).chunk(2, dim=-1)

        attended = _multi_head_attention(query, key, value, self.num_heads, mask)
        x = x + self.to_output(attended)
        return x + self.mlp(self.norm_mlp(x))


def _multi_head_attention(query: Tensor, key: Tensor, value: Tensor, num_heads: int, mask: Tensor | None) -> Tensor:
    batch_size, query_length, width = query.shape
    head_width = width // num_heads

    def split_heads(tokens: Tensor) -> Tensor:  # (N, L, width) -> (N, heads, L, head_width)
        return tokens.reshape(batch_size, -1, num_heads, head_width).transpose(1, 2)

    q, k, v = split_heads(query), split_heads(key), split_heads(value)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_width)  # (N, heads, Lq, Lk)
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
    attended = scores.softmax(dim=-1) @ v
    return attended.transpose(1, 2).reshape(batch_size, query_length, width)


class HieraRayTransformer(nn.Module):
    """Processes the samples of each ray together, at two scales, in the manner of Hiera.

    1. Local: each sample attends to the samples in its group of
       ``window_size`` consecutive samples.
    2. Merge: neighbouring pairs of samples are merged into one token
       (a learnt linear map of the pair), halving the length of the ray.
    3. Global: the merged tokens attend across the whole, shorter ray.
    4. Fuse: every merged token is copied back to the two samples it came from
       and combined with their local tokens, so each sample gets its own output.

    Rays of any length work: the samples are padded to a whole number of
    groups, and padding is never attended to.
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        window_size: int = 8,
        local_depth: int = 1,
        global_depth: int = 1,
        mlp_ratio: float = 2.0,
    ) -> None:
        """
        Args:
            width: Channels of every token.
            num_heads: Attention heads.
            window_size: Samples per local group; an even number.
            local_depth: Attention blocks within the groups.
            global_depth: Attention blocks across the merged ray.
            mlp_ratio: Hidden width of the blocks' MLPs, relative to ``width``.
        """
        super().__init__()
        if window_size < 2 or window_size % 2:
            raise ValueError(f"window_size must be an even number of at least 2, got {window_size}.")
        if local_depth < 1 or global_depth < 1:
            raise ValueError("local_depth and global_depth must be at least 1.")
        self.window_size = window_size
        self.local_blocks = nn.ModuleList(
            AttentionBlock(width, num_heads, mlp_ratio) for _ in range(local_depth)
        )
        self.merge_norm = nn.LayerNorm(2 * width)
        self.merge = nn.Linear(2 * width, width)
        self.global_blocks = nn.ModuleList(
            AttentionBlock(width, num_heads, mlp_ratio) for _ in range(global_depth)
        )
        self.fuse_norm_local = nn.LayerNorm(width)
        self.fuse_norm_global = nn.LayerNorm(width)
        self.fuse = nn.Linear(2 * width, width)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: Shape (N, S, width): N rays with S samples each, ordered near to far.

        Returns:
            Shape (N, S, width).
        """
        num_rays, num_samples, width = tokens.shape
        window = self.window_size
        num_windows = -(-num_samples // window)
        padded_length = num_windows * window

        # Padding only ever fills the last group partially, so every group
        # keeps at least one real sample to attend to.
        is_real = torch.arange(padded_length, device=tokens.device) < num_samples
        x = F.pad(tokens, (0, 0, 0, padded_length - num_samples))

        # 1. Local attention within groups.
        groups = x.reshape(num_rays * num_windows, window, width)
        group_mask = is_real.reshape(1, num_windows, window).expand(num_rays, -1, -1).reshape(-1, window)
        for block in self.local_blocks:
            groups = block(groups, mask=group_mask)
        local = groups.reshape(num_rays, padded_length, width) * is_real[None, :, None]

        # 2. Merge neighbouring pairs.
        merged = self.merge(self.merge_norm(local.reshape(num_rays, padded_length // 2, 2 * width)))
        merged_is_real = is_real.reshape(-1, 2).any(dim=-1)

        # 3. Global attention across the shorter ray.
        merged_mask = merged_is_real[None].expand(num_rays, -1)
        for block in self.global_blocks:
            merged = block(merged, mask=merged_mask)

        # 4. Fuse the scales back, one output per sample.
        upsampled = merged.repeat_interleave(2, dim=1)
        fused = local + self.fuse(
            torch.cat([self.fuse_norm_local(local), self.fuse_norm_global(upsampled)], dim=-1)
        )
        return fused[:, :num_samples]


def sinusoidal_encoding(values: Tensor, num_frequencies: int) -> Tensor:
    """Append sines and cosines of the input at octave-spaced frequencies (NeRF, Sec. 5.1).

    Args:
        values: Shape (..., D).
        num_frequencies: Number of frequencies L.

    Returns:
        Shape (..., D + D * 2 * L): the input followed by its encodings.
    """
    frequencies = math.pi * 2.0 ** torch.arange(num_frequencies, device=values.device, dtype=values.dtype)
    scaled = values[..., None] * frequencies  # (..., D, L)
    encodings = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-2)
    return torch.cat([values, encodings], dim=-1)


def encoding_size(num_inputs: int, num_frequencies: int) -> int:
    """Length of ``sinusoidal_encoding`` of ``num_inputs`` values."""
    return num_inputs * (1 + 2 * num_frequencies)
