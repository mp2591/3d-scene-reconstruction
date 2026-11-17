"""A three-layer convolutional image encoder. For testing only."""

from __future__ import annotations

from torch import Tensor, nn

from scene_recon.models.normalization import InputNormalization
from scene_recon.registry import ENCODERS


@ENCODERS.register("tiny_conv")
class TinyConvEncoder(nn.Module):
    """Maps an image to a feature map at half its resolution."""

    def __init__(self, num_features: int = 16, input_normalization: str = "none") -> None:
        """
        Args:
            num_features: Channels of the output feature map.
            input_normalization: How input images are normalised; see
                ``scene_recon.models.normalization``. This encoder is trained
                from scratch, so it needs none; a pretrained one would use
                the normalisation of its weights, e.g. ``"imagenet"``.
        """
        super().__init__()
        self.num_features = num_features
        self.normalize = InputNormalization(input_normalization)
        self.layers = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
        )

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: Shape (N, 3, H, W) in [0, 1].

        Returns:
            Feature maps of shape (N, num_features, H / 2, W / 2).
        """
        return self.layers(self.normalize(images))
