from typing import Any

import pytest
from torch import Tensor, nn

from scene_recon.config import OptimizationSettings
from scene_recon.models.base import ParameterGroup, ReconstructionModel
from scene_recon.optim import build_optimizer_and_scheduler, learning_rate_factor
from scene_recon.registry import ConfigError
from scene_recon.structures import FieldOutput, SceneViews


class _ThreeGroupModel(ReconstructionModel):
    """A stand-in with a two-stage encoder and a decoder; only its parameter groups matter here."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder_stage0 = nn.Linear(2, 2)
        self.encoder_stage1 = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)

    def encode_scene(self, source_views: SceneViews) -> Any:
        raise NotImplementedError

    def query_field(self, scene_encoding: Any, points: Tensor, view_directions: Tensor, level: str) -> FieldOutput:
        raise NotImplementedError

    def parameter_groups(self) -> list[ParameterGroup]:
        return [
            ParameterGroup("encoder.stage0", list(self.encoder_stage0.parameters()), lr_scale=0.5),
            ParameterGroup("encoder.stage1", list(self.encoder_stage1.parameters())),
            ParameterGroup("decoder", list(self.decoder.parameters())),
        ]


def _settings(**changes: Any) -> OptimizationSettings:
    values = dict(
        optimizer="adam",
        betas=[0.9, 0.999],
        weight_decay=0.0,
        learning_rate=1e-4,
        warmup_steps=0,
        decay_rate=1.0,
        decay_steps=1000,
        group_overrides={},
    )
    values.update(changes)
    return OptimizationSettings(**values)


def _learning_rates(optimizer) -> dict[str, float]:
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


def test_factor_combines_linear_warmup_with_exponential_decay():
    assert learning_rate_factor(step=0, warmup_steps=0, decay_rate=1.0, decay_steps=100) == 1.0
    assert learning_rate_factor(step=4, warmup_steps=10, decay_rate=1.0, decay_steps=100) == pytest.approx(0.5)
    assert learning_rate_factor(step=100, warmup_steps=0, decay_rate=0.1, decay_steps=100) == pytest.approx(0.1)
    assert learning_rate_factor(step=50, warmup_steps=0, decay_rate=0.1, decay_steps=100) == pytest.approx(0.1**0.5)


def test_every_group_gets_the_default_learning_rate_times_its_scale():
    optimizer, _ = build_optimizer_and_scheduler(_ThreeGroupModel(), _settings())

    assert _learning_rates(optimizer) == pytest.approx(
        {"encoder.stage0": 0.5e-4, "encoder.stage1": 1e-4, "decoder": 1e-4}
    )


def test_override_applies_to_a_group_and_everything_nested_under_it():
    settings = _settings(group_overrides={"encoder": {"learning_rate": 1e-5}})

    optimizer, _ = build_optimizer_and_scheduler(_ThreeGroupModel(), settings)

    assert _learning_rates(optimizer) == pytest.approx(
        {"encoder.stage0": 0.5e-5, "encoder.stage1": 1e-5, "decoder": 1e-4}
    )


def test_most_specific_override_wins():
    settings = _settings(
        group_overrides={"encoder": {"learning_rate": 1e-5}, "encoder.stage1": {"learning_rate": 3e-5}}
    )

    optimizer, _ = build_optimizer_and_scheduler(_ThreeGroupModel(), settings)

    assert _learning_rates(optimizer)["encoder.stage1"] == pytest.approx(3e-5)


def test_warmup_can_be_limited_to_one_group():
    settings = _settings(group_overrides={"encoder": {"warmup_steps": 10}})
    optimizer, scheduler = build_optimizer_and_scheduler(_ThreeGroupModel(), settings)

    rates_at_start = _learning_rates(optimizer)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    rates_after_warmup = _learning_rates(optimizer)

    assert rates_at_start["encoder.stage1"] == pytest.approx(1e-5)  # one tenth of the way through warmup
    assert rates_at_start["decoder"] == pytest.approx(1e-4)  # no warmup
    assert rates_after_warmup["encoder.stage1"] == pytest.approx(1e-4)


def test_override_for_a_group_the_model_does_not_have_is_rejected():
    settings = _settings(group_overrides={"backbone": {"learning_rate": 1e-5}})

    with pytest.raises(ConfigError, match="backbone"):
        build_optimizer_and_scheduler(_ThreeGroupModel(), settings)


def test_override_of_an_unsupported_setting_is_rejected():
    settings = _settings(group_overrides={"encoder": {"momentum": 0.9}})

    with pytest.raises(ConfigError, match="momentum"):
        build_optimizer_and_scheduler(_ThreeGroupModel(), settings)
