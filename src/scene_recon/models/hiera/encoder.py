"""The Hiera-T image encoder, with its first three stages fused into one feature map.

Hiera (Ryali et al., "Hiera: A Hierarchical Vision Transformer without the
Bells-and-Whistles", ICML 2023) is a vision transformer with no convolutions
apart from the patch embedding. The photo is cut into 4x4-pixel tokens that
pass through four stages of transformer blocks. In the first two stages each
token attends only to the tokens of its own mask unit (an 8x8-token, i.e.
32x32-pixel, square); the later stages attend across the whole photo.
Between stages the token grid halves in height and width (by max-pooling the
queries) while each token doubles its number of channels. Hiera-T was
pretrained as a masked autoencoder on ImageNet: 60% of the mask units of each
photo were hidden and the network learnt to fill them in.

This file re-implements the Hiera encoder with the module names of the
official code (``facebookresearch/hiera``, which ``timm`` also uses), so
official checkpoints load directly. Only the first three stages are built:
their outputs (96, 192 and 384 channels at 1/4, 1/8 and 1/16 of the photo's
height and width) are what the model uses, and the fourth stage would be
parameters that never influence the output.

The three stage outputs are resized to the 1/4 grid, stacked (672 channels)
and turned into ``num_features`` channels by a 1x1 convolution, the adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scene_recon.models.normalization import InputNormalization
from scene_recon.registry import ENCODERS, ConfigError

_PATCH_STRIDE = 4
"""Pixels per token along each axis."""

_Q_STRIDE = 2
"""How much the token grid shrinks along each axis between stages."""

_NUM_UNROLLS = 3
"""Number of unroll steps. The full four-stage Hiera unrolls once per stage
transition; it is kept at 3 although stage 4 is not built, because the token
order it produces is what the mask-unit attention of the pretrained weights
works on."""

_MASK_UNIT_TOKENS = 8
"""Side of a mask unit in tokens at the first stage (8 tokens = 32 pixels)."""

_LAYER_NORM_EPS = 1e-6

IMAGE_SIZE_MULTIPLE = _PATCH_STRIDE * _Q_STRIDE**_NUM_UNROLLS
"""Image height and width must be multiples of this (32 pixels)."""


class DropPath(nn.Module):
    """Stochastic depth: drops a whole residual branch for random images during training."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MaskUnitAttention(nn.Module):
    """Multi-head attention, either within each mask unit or across all tokens.

    The tokens arrive "unrolled" (see ``unroll``), so the tokens of one mask
    unit are those whose index has the same remainder modulo the number of
    mask units, and max-pooling the queries over the leading quarter-blocks
    of the sequence is a 2x2 max-pool in image space.
    """

    def __init__(
        self, dim: int, dim_out: int, heads: int, q_stride: int, window_size: int, use_mask_unit_attn: bool
    ) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.heads = heads
        self.head_dim = dim_out // heads
        self.q_stride = q_stride
        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn
        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, _ = x.shape
        num_windows = num_tokens // (self.q_stride * self.window_size) if self.use_mask_unit_attn else 1

        qkv = self.qkv(x).reshape(batch_size, -1, num_windows, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(3, 0, 4, 2, 1, 5)  # each (B, heads, windows, tokens per window, head_dim)
        if self.q_stride > 1:
            q = q.reshape(batch_size, self.heads, num_windows, self.q_stride, -1, self.head_dim).max(dim=3).values

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 3).reshape(batch_size, -1, self.dim_out)
        return self.proj(x)


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float,
        drop_path: float,
        q_stride: int,
        window_size: int,
        use_mask_unit_attn: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = nn.LayerNorm(dim, eps=_LAYER_NORM_EPS)
        self.attn = MaskUnitAttention(dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn)
        self.norm2 = nn.LayerNorm(dim_out, eps=_LAYER_NORM_EPS)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            # The residual path is projected and pooled like the queries.
            x = _max_pool_tokens(self.proj(x_norm), self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim_out, kernel_size=7, stride=_PATCH_STRIDE, padding=3)

    def forward(self, images: Tensor) -> Tensor:
        return self.proj(images)  # (N, C, H / 4, W / 4)


class HieraBackbone(nn.Module):
    """The first three stages of Hiera. Its parameter names match the official checkpoints."""

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        stage_depths: Sequence[int] = (1, 2, 7),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        pretrain_image_size: int = 224,
    ) -> None:
        super().__init__()
        if len(stage_depths) != 3:
            raise ConfigError(f"stage_depths must list the depths of Hiera's first three stages, got {stage_depths}.")
        mask_unit_attention = (True, True, False)
        flat_q_stride = _Q_STRIDE**2

        self.stage_ends = [sum(stage_depths[: index + 1]) - 1 for index in range(3)]
        q_pool_blocks = [end + 1 for end in self.stage_ends[:2]]
        depth = sum(stage_depths)

        self.patch_embed = PatchEmbed(embed_dim)
        self.pretrain_grid = pretrain_image_size // _PATCH_STRIDE
        self.pos_embed = nn.Parameter(torch.zeros(1, self.pretrain_grid**2, embed_dim))

        drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        flat_mask_unit_size = _MASK_UNIT_TOKENS**2
        dim, heads, stage = embed_dim, num_heads, 0
        self.blocks = nn.ModuleList()
        self.stage_channels = []
        for index in range(depth):
            dim_out = dim
            # As in the official code, the kind of attention lags one block
            # behind the stage: the first block of a stage pools its queries
            # and still attends within mask units like the previous stage.
            use_mask_unit_attn = mask_unit_attention[stage]
            if index - 1 in self.stage_ends:
                dim_out = dim * 2
                heads = heads * 2
                stage += 1
                if index in q_pool_blocks:
                    flat_mask_unit_size //= flat_q_stride
            self.blocks.append(
                HieraBlock(
                    dim=dim,
                    dim_out=dim_out,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rates[index],
                    q_stride=flat_q_stride if index in q_pool_blocks else 1,
                    window_size=flat_mask_unit_size,
                    use_mask_unit_attn=use_mask_unit_attn,
                )
            )
            dim = dim_out
            if index in self.stage_ends:
                self.stage_channels.append(dim)
        self._pools_before_block = [sum(block <= index for block in q_pool_blocks) for index in range(depth)]

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_weights)

    def forward(self, images: Tensor) -> list[Tensor]:
        """
        Args:
            images: Normalised images of shape (N, 3, H, W); H and W multiples of 32.

        Returns:
            The outputs of the three stages, of shapes (N, 96, H/4, W/4),
            (N, 192, H/8, W/8) and (N, 384, H/16, W/16) for Hiera-T.
        """
        tokens = self.patch_embed(images)
        grid = (tokens.shape[-2], tokens.shape[-1])
        tokens = tokens.flatten(2).transpose(1, 2) + self._position_embedding(grid)
        tokens = unroll(tokens, grid)

        outputs = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in self.stage_ends:
                outputs.append(reroll(tokens, grid, self._pools_before_block[index]))
        return outputs

    def _position_embedding(self, grid: tuple[int, int]) -> Tensor:
        """The learnt position embedding, resized from the pretraining grid if needed."""
        if grid == (self.pretrain_grid, self.pretrain_grid):
            return self.pos_embed
        embedding = self.pos_embed.reshape(1, self.pretrain_grid, self.pretrain_grid, -1).permute(0, 3, 1, 2)
        embedding = F.interpolate(embedding, size=grid, mode="bicubic", align_corners=False)
        return embedding.flatten(2).transpose(1, 2)


@ENCODERS.register("hiera")
class HieraEncoder(nn.Module):
    """Hiera-T, with its three stage outputs fused into one feature map per image.

    The output has ``num_features`` channels at 1/4 of the image's height and
    width. Image height and width must be multiples of 32 (128x128
    ShapeNet-SRN images give a 32x32 feature map).

    Parameters are split for the optimiser: ``backbone`` holds the pretrained
    Hiera weights, ``adapter`` the 1x1 convolution that is trained from scratch.
    """

    def __init__(
        self,
        weights: str | None = None,
        num_features: int = 256,
        input_normalization: str = "imagenet",
        embed_dim: int = 96,
        num_heads: int = 1,
        stage_depths: Sequence[int] = (1, 2, 7),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        pretrain_image_size: int = 224,
    ) -> None:
        """
        Args:
            weights: Path to a Hiera checkpoint (see ``docs/hiera_nerf.md``), or
                ``None`` to start from random weights (tests only).
            num_features: Channels of the output feature map.
            input_normalization: How images are normalised first. The official
                Hiera weights expect ``"imagenet"``.
            embed_dim: Channels of the first stage (96 for Hiera-T).
            num_heads: Attention heads of the first stage (1 for Hiera-T);
                doubled at each stage.
            stage_depths: Blocks in each of the first three stages ((1, 2, 7) for Hiera-T).
            mlp_ratio: Hidden width of each block's MLP, relative to its channels.
            drop_path_rate: Stochastic depth rate of the last block, rising linearly from 0.
            pretrain_image_size: Image size of the pretraining, which fixes the
                size of the stored position embedding (224 for the official weights).
        """
        super().__init__()
        self.num_features = num_features
        self.normalize = InputNormalization(input_normalization)
        self.backbone = HieraBackbone(
            embed_dim=embed_dim,
            num_heads=num_heads,
            stage_depths=stage_depths,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            pretrain_image_size=pretrain_image_size,
        )
        if weights is not None:
            load_hiera_weights(self.backbone, weights)
        self.adapter = nn.Conv2d(sum(self.backbone.stage_channels), num_features, kernel_size=1)

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: Shape (N, 3, H, W), values in [0, 1]; H and W multiples of 32.

        Returns:
            Feature maps of shape (N, num_features, H / 4, W / 4).
        """
        height, width = images.shape[-2:]
        if height % IMAGE_SIZE_MULTIPLE or width % IMAGE_SIZE_MULTIPLE:
            raise ValueError(
                f"The Hiera encoder needs image sides that are multiples of {IMAGE_SIZE_MULTIPLE} pixels "
                f"(whole mask units), got {height}x{width}."
            )
        stages = self.backbone(self.normalize(images))
        size = stages[0].shape[-2:]
        resized = [stages[0]] + [
            F.interpolate(stage, size=size, mode="bilinear", align_corners=False) for stage in stages[1:]
        ]
        return self.adapter(torch.cat(resized, dim=1))


# ------------------------------------------------------------------ token order


def unroll(tokens: Tensor, grid: tuple[int, int]) -> Tensor:
    """Reorder tokens from row-major order into Hiera's "unrolled" order.

    Each step moves the position within every 2x2 block to the front of the
    sequence. After the steps, token ``n`` has index
    ``n = s1 * N/4 + s2 * N/16 + s3 * N/64 + (row-major index in the 1/8 grid)``,
    where ``s1, s2, s3`` are its positions within the nested 2x2 blocks. So:

    * max-pooling over the four quarters of the sequence is a 2x2 max-pool of
      the image, and the result is unrolled the same way, one step fewer;
    * the tokens of one 8x8 mask unit are those with the same index modulo N/64.

    Args:
        tokens: Shape (B, H * W, C) in row-major order over ``grid``.
        grid: (H, W), both multiples of 8.

    Returns:
        Shape (B, H * W, C).
    """
    batch_size, num_tokens, channels = tokens.shape
    height, width = grid
    x = tokens.reshape(batch_size, height, width, channels)
    for _ in range(_NUM_UNROLLS):
        height, width = height // _Q_STRIDE, width // _Q_STRIDE
        x = x.reshape(-1, height, _Q_STRIDE, width, _Q_STRIDE, channels)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(0, 2)  # the 2x2 position joins the batch dimension
    return x.reshape(batch_size, num_tokens, channels)


def reroll(tokens: Tensor, grid: tuple[int, int], num_pools: int) -> Tensor:
    """Turn unrolled tokens back into an image-shaped feature map.

    Args:
        tokens: Shape (B, N, C), unrolled, after ``num_pools`` 2x2 poolings
            of a token grid of size ``grid`` (so N = H * W / 4**num_pools).
        grid: (H, W) of the token grid before any pooling.
        num_pools: Number of poolings so far.

    Returns:
        Shape (B, C, H / 2**num_pools, W / 2**num_pools).
    """
    batch_size, num_tokens, channels = tokens.shape
    scale = _Q_STRIDE**num_pools
    height, width = grid[0] // scale, grid[1] // scale
    # After pooling, token j stands for the block of the original token
    # unrolled to position j (the one with s1 = ... = s_k = 0).
    original = token_order(grid, tokens.device)[:num_tokens]
    cell = (original // grid[1]) // scale * width + (original % grid[1]) // scale
    feature_map = tokens[:, torch.argsort(cell)]
    return feature_map.transpose(1, 2).reshape(batch_size, channels, height, width)


def token_order(grid: tuple[int, int], device: torch.device | str | None = None) -> Tensor:
    """For each unrolled position, the row-major index of the token found there. Shape (H * W,)."""
    num_tokens = grid[0] * grid[1]
    indices = torch.arange(num_tokens, device=device).view(1, num_tokens, 1)
    return unroll(indices, grid).reshape(num_tokens)


def _max_pool_tokens(x: Tensor, stride: int) -> Tensor:
    """2x2 max-pool of unrolled tokens: the maximum over the four quarters of the sequence."""
    return x.reshape(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


def _init_weights(module: nn.Module) -> None:
    """The initialisation of the official code, used when no checkpoint is loaded."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0.02)
        nn.init.constant_(module.weight, 1.0)


# ------------------------------------------------------------------ pretrained weights


def load_hiera_weights(backbone: HieraBackbone, path: str | Path) -> None:
    """Load official Hiera weights into the first three stages.

    Accepts a plain state dict (``model.state_dict()`` of the official or the
    ``timm`` Hiera) or an official checkpoint file, which stores it under
    ``"model_state"``. Keys the backbone does not have (the fourth stage, the
    classification head, the masked-autoencoder decoder) are ignored. Every
    key the backbone does have must be present with the right shape.

    Raises:
        ConfigError: If the file is missing or does not fit the backbone.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Hiera weights not found: {path}. See docs/hiera_nerf.md for how to get them.")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - any unpickling failure gets the same advice
        raise ConfigError(
            f"Could not read Hiera weights from {path} ({error}). Save the bare state dict instead: "
            "torch.save(model.state_dict(), path)."
        ) from error

    state = checkpoint
    for key in ("model_state", "state_dict", "model"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}

    expected = backbone.state_dict()
    missing = [key for key in expected if key not in state]
    if missing:
        raise ConfigError(
            f"The Hiera weights in {path} lack {len(missing)} entries the encoder needs "
            f"(e.g. {', '.join(missing[:3])}). Are they Hiera-T weights?"
        )
    mismatched = [
        f"{key}: {tuple(state[key].shape)} instead of {tuple(value.shape)}"
        for key, value in expected.items()
        if state[key].shape != value.shape
    ]
    if mismatched:
        raise ConfigError(
            f"The Hiera weights in {path} do not fit the encoder's shape ({'; '.join(mismatched[:3])}). "
            "Check embed_dim, num_heads, stage_depths and pretrain_image_size."
        )
    backbone.load_state_dict({key: state[key] for key in expected})

