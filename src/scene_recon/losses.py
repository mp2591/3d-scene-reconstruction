"""Training losses.

A loss is a callable that receives the renders of every pass together with
the ground-truth colours, and returns a dictionary of named scalar terms. The
entry ``"total"`` is the one that is back-propagated; the others are logged.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from scene_recon.models.base import LEVEL_COARSE, LEVEL_FINE
from scene_recon.registry import LOSSES
from scene_recon.structures import RenderOutput


@LOSSES.register("mse")
class PhotometricMSELoss:
    """Mean squared error between rendered and true pixel colours.

    Both the coarse and the fine render are supervised, as in NeRF: the fine
    pass can only place its samples well if the coarse pass is accurate too.
    """

    def __init__(self, coarse_weight: float = 1.0, fine_weight: float = 1.0) -> None:
        self.level_weights = {LEVEL_COARSE: coarse_weight, LEVEL_FINE: fine_weight}

    def __call__(self, renders: dict[str, RenderOutput], target_rgb: Tensor) -> dict[str, Tensor]:
        """
        Args:
            renders: One ``RenderOutput`` per rendering pass, keyed by level name.
            target_rgb: Ground-truth colours of shape (B, R, 3).
        """
        terms = {f"mse_{level}": F.mse_loss(render.rgb, target_rgb) for level, render in renders.items()}
        terms["total"] = sum(self.level_weights[level] * terms[f"mse_{level}"] for level in renders)
        return terms
