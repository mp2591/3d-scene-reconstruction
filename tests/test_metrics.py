import pytest
import torch

from scene_recon.build import build_metrics
from scene_recon.metrics import PSNR, SSIM
from scene_recon.registry import ConfigError


def test_psnr_matches_its_definition():
    target = torch.zeros(2, 3, 8, 8)
    prediction = torch.full((2, 3, 8, 8), 0.1)  # mean squared error of 0.01

    values = PSNR()(prediction, target)

    assert torch.allclose(values, torch.tensor([20.0, 20.0]), atol=1e-4)


def test_psnr_of_identical_images_is_finite():
    image = torch.rand(1, 3, 8, 8)

    assert torch.isfinite(PSNR()(image, image)).all()


def test_ssim_is_one_for_identical_images_and_lower_for_noisy_ones():
    image = torch.rand(2, 3, 16, 16)
    noisy = (image + 0.2 * torch.randn_like(image)).clamp(0, 1)

    assert torch.allclose(SSIM()(image, image), torch.ones(2), atol=1e-6)
    assert (SSIM()(noisy, image) < 0.99).all()


def test_metrics_are_built_by_name_in_the_listed_order():
    metrics = build_metrics(["ssim", "psnr"])

    assert list(metrics) == ["ssim", "psnr"]


def test_unknown_metric_is_a_config_error():
    with pytest.raises(ConfigError, match="Unknown metric"):
        build_metrics(["sharpness"])
