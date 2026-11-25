"""A ConvNeXt V2 image encoder that produces one feature map per image."""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scene_recon.models.base import describe_pretrained_weights
from scene_recon.models.normalization import InputNormalization
from scene_recon.registry import ENCODERS


@ENCODERS.register("convnextv2")
class ConvNeXtV2Encoder(nn.Module):
    """Turns each image into one feature map, using a pretrained ConvNeXt V2.

    ConvNeXt V2 sees the image at four zoom levels. Their outputs are resized to
    the same size, stacked, and squeezed to ``num_features`` channels by a 1x1
    convolution, the ``adapter``, which is trained from scratch.
    """

    feature_offset = 0.0
    """The stem (4x4 convolution, stride 4) tiles the image evenly; see ``sample_features_at_pixels``."""

    def __init__(
        self,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        num_features: int = 256,
        input_normalization: str = "imagenet",
    ) -> None:
        """
        Args:
            model_name: The ``timm`` name of the ConvNeXt V2 variant and its weights.
            pretrained: Start from pretrained weights. Use False for quick tests.
            num_features: Channels of the output feature map.
            input_normalization: How images are normalised first. ConvNeXt V2 expects
                "imagenet".
        """
        super().__init__()
        self.num_features = num_features
        self.normalize = InputNormalization(input_normalization)

        # features_only=True drops the classification head and returns the
        # output of each of the four stages instead.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.pretrained_weights = describe_pretrained_weights(f"timm {model_name}", self.backbone) if pretrained else None
        total_channels = sum(self.backbone.feature_info.channels())
        self.adapter = nn.Conv2d(total_channels, num_features, kernel_size=1)
        # The stages line up with each other and with feature_offset only when
        # each one tiles the image evenly, so image sides must be multiples of
        # the coarsest stage's stride (32 for every ConvNeXt V2).
        self.image_size_multiple = max(self.backbone.feature_info.reduction())

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: Shape (N, 3, H, W), values in [0, 1]; H and W multiples of 32.

        Returns:
            Feature maps of shape (N, num_features, H / 4, W / 4).
        """
        height, width = images.shape[-2:]
        if height % self.image_size_multiple or width % self.image_size_multiple:
            raise ValueError(
                f"The ConvNeXt V2 encoder needs image sides that are multiples of {self.image_size_multiple} "
                f"pixels (the stride of its coarsest stage), got {height}x{width}."
            )
        stages = self.backbone(self.normalize(images))
        size = stages[0].shape[-2:]  # the largest feature map
        resized = [stages[0]] + [
            F.interpolate(stage, size=size, mode="bilinear", align_corners=False)
            for stage in stages[1:]
        ]
        return self.adapter(torch.cat(resized, dim=1))  # (N, num_features, H/4, W/4)
