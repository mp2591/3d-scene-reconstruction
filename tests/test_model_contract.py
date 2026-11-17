from typing import Any

import pytest
import torch
from torch import Tensor, nn

from scene_recon.build import build_model
from scene_recon.config import Config
from scene_recon.models.base import ParameterGroup, ReconstructionModel
from scene_recon.models.contract import ModelContractError, check_model_contract
from scene_recon.structures import FieldOutput, SceneViews


class _MinimalModel(ReconstructionModel):
    """The smallest model that satisfies the contract. The tests below break it in one way each."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, 4)

    def encode_scene(self, source_views: SceneViews) -> Any:
        return None

    def query_field(self, scene_encoding: Any, points: Tensor, view_directions: Tensor, level: str) -> FieldOutput:
        output = self.layer(points)
        return FieldOutput(rgb=torch.sigmoid(output[..., :3]), density=torch.relu(output[..., 3]))


def test_minimal_model_passes():
    check_model_contract(_MinimalModel())


def test_testing_model_passes(smoke_config: Config):
    check_model_contract(build_model(smoke_config))


def test_colours_outside_zero_to_one_are_reported():
    class UnboundedColour(_MinimalModel):
        def query_field(self, scene_encoding, points, view_directions, level):
            field = super().query_field(scene_encoding, points, view_directions, level)
            return FieldOutput(rgb=field.rgb * 5.0, density=field.density)

    with pytest.raises(ModelContractError, match=r"rgb must lie in \[0, 1\]"):
        check_model_contract(UnboundedColour())


def test_negative_density_is_reported():
    class NegativeDensity(_MinimalModel):
        def query_field(self, scene_encoding, points, view_directions, level):
            field = super().query_field(scene_encoding, points, view_directions, level)
            return FieldOutput(rgb=field.rgb, density=field.density - 1.0)

    with pytest.raises(ModelContractError, match="density must be non-negative"):
        check_model_contract(NegativeDensity())


def test_wrong_output_shape_is_reported():
    class FlattenedOutput(_MinimalModel):
        def query_field(self, scene_encoding, points, view_directions, level):
            field = super().query_field(scene_encoding, points, view_directions, level)
            return FieldOutput(rgb=field.rgb.flatten(1, 2), density=field.density.flatten(1, 2))

    with pytest.raises(ModelContractError, match="rgb has shape"):
        check_model_contract(FlattenedOutput())


def test_parameters_missing_from_every_group_are_reported():
    class ForgottenParameters(_MinimalModel):
        def __init__(self) -> None:
            super().__init__()
            self.forgotten = nn.Linear(2, 2)

        def parameter_groups(self) -> list[ParameterGroup]:
            return [ParameterGroup("layer", list(self.layer.parameters()))]

    with pytest.raises(ModelContractError, match="in no parameter group"):
        check_model_contract(ForgottenParameters())
