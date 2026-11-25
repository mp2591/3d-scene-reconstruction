"""Image quality metrics for novel-view synthesis.

Every metric is an ``nn.Module`` that compares a batch of rendered images
with the ground truth and returns one value per image::

    values = metric(prediction, target)   # both (N, 3, H, W) in [0, 1] -> (N,)

and declares three class attributes used by the suite report:
``display_name``, ``higher_is_better`` and ``software``, the packages besides
PyTorch whose code computes it. Every evaluation records their versions, and
the report says when runs were scored with different ones.

SSIM and LPIPS follow the settings of the pixelNeRF evaluation code so that
numbers are comparable with the published baselines.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from scene_recon.registry import METRICS, ConfigError


@METRICS.register("psnr")
class PSNR(nn.Module):
    """Peak signal-to-noise ratio in decibels. Higher is better."""

    display_name = "PSNR (dB)"
    higher_is_better = True
    software = ()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        mse = ((prediction - target) ** 2).mean(dim=(1, 2, 3))
        return mse_to_psnr(mse)


@METRICS.register("ssim")
class SSIM(nn.Module):
    """Structural similarity (Wang et al., 2004), in [-1, 1]. Higher is better.

    Computed with scikit-image's default settings (a 7x7 uniform window),
    which is what the pixelNeRF and TransNeRF evaluations use.
    """

    display_name = "SSIM"
    higher_is_better = True
    software = ("scikit-image",)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        from skimage.metrics import structural_similarity

        prediction_images = prediction.detach().permute(0, 2, 3, 1).cpu().numpy()
        target_images = target.detach().permute(0, 2, 3, 1).cpu().numpy()
        values = [
            structural_similarity(predicted, true, channel_axis=-1, data_range=1.0)
            for predicted, true in zip(prediction_images, target_images)
        ]
        return torch.tensor(values, dtype=torch.float32, device=prediction.device)


@METRICS.register("lpips")
class LPIPS(nn.Module):
    """Learned perceptual image patch similarity (Zhang et al., 2018). Lower is better.

    Requires the optional ``lpips`` package (``pip install scene-recon[lpips]``),
    which downloads pretrained network weights on first use.
    """

    display_name = "LPIPS"
    higher_is_better = False
    software = ("lpips", "torchvision")  # lpips runs torchvision's VGG

    def __init__(self, network: str = "vgg") -> None:
        """
        Args:
            network: Backbone used to compare images: ``"vgg"`` (used by the
                baselines) or ``"alex"``.
        """
        super().__init__()
        try:
            import lpips
        except ImportError as error:
            raise ConfigError(
                "The 'lpips' metric needs the lpips package. Install it with "
                "`pip install lpips`, or remove 'lpips' from the metrics list in your config."
            ) from error
        self.network = lpips.LPIPS(net=network, verbose=False)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        # The lpips package expects images in [-1, 1].
        distance = self.network(prediction * 2.0 - 1.0, target * 2.0 - 1.0)
        return distance.reshape(-1)


def mse_to_psnr(mse: Tensor) -> Tensor:
    """Convert mean squared error of images in [0, 1] to PSNR in decibels."""
    # The clamp caps the PSNR of a pixel-perfect image at 100 dB instead of infinity.
    return -10.0 * torch.log10(mse.clamp(min=1e-10))
