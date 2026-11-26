"""Scoring as pixelNeRF scores: random sample positions that can be reproduced, and 8-bit renders."""

import torch
from torch import nn

from scene_recon.build import build_model
from scene_recon.config import Config
from scene_recon.data.synthetic import SyntheticSpheresDataset
from scene_recon.evaluation import Evaluator
from scene_recon.metrics import PSNR
from scene_recon.rendering import VolumeRenderer

CPU = torch.device("cpu")


class _RecordingRenderer(VolumeRenderer):
    """Renders as usual, and keeps the seed of each generator it is given."""

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.seeds: list[int] = []

    def render_images(self, model, scene_encoding, target_views, generator=None):
        assert generator is not None, "evaluation must draw its sample positions at random, as pixelNeRF does"
        self.seeds.append(generator.initial_seed())
        return super().render_images(model, scene_encoding, target_views, generator)


def test_random_sample_positions_reproduce_whatever_the_number_of_scenes(smoke_config: Config):
    model = build_model(smoke_config)
    scenes = SyntheticSpheresDataset(num_scenes=2, num_views=3, image_size=8, seed=0)

    def score(num_scenes: int) -> tuple[dict, list[int]]:
        renderer = _RecordingRenderer(smoke_config.renderer)
        evaluator = Evaluator(model, renderer, {"psnr": PSNR()}, CPU, seed=0)
        result = evaluator.evaluate(scenes, [0], num_scenes=num_scenes, num_preview_scenes=0)
        return result.per_scene, renderer.seeds

    one_scene, one_seed = score(1)
    two_scenes, two_seeds = score(2)

    assert two_seeds[0] == one_seed[0] and two_seeds[1] != two_seeds[0]  # one stream per scene
    first = next(iter(one_scene))
    assert two_scenes[first] == one_scene[first]


class _ConstantRenderer(VolumeRenderer):
    """Renders every pixel at 0.3, between two 8-bit levels (76.5 / 255)."""

    def render_images(self, model, scene_encoding, target_views, generator=None):
        height, width = target_views.image_size
        shape = (target_views.num_scenes, target_views.num_views)
        return torch.full((*shape, 3, height, width), 0.3), torch.zeros(*shape, height, width)


class _RecordingMetric(nn.Module):
    def forward(self, prediction, truth):
        self.prediction = prediction
        return prediction.new_zeros(prediction.shape[0])


def test_renders_are_scored_as_8_bit_images_rounded_down(smoke_config: Config):
    metric = _RecordingMetric()
    evaluator = Evaluator(build_model(smoke_config), _ConstantRenderer(smoke_config.renderer), {"m": metric}, CPU, 0)

    evaluator.evaluate(SyntheticSpheresDataset(1, 3, 8, 0), [0], num_preview_scenes=0)

    # pixelNeRF saves (rgb * 255).astype(uint8), which rounds 76.5 down to 76, and scores that.
    assert torch.equal(metric.prediction, torch.full_like(metric.prediction, 76 / 255))
