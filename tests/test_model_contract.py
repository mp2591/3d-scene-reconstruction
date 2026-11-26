import random
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from scene_recon.build import build_model, build_seeded_model
from scene_recon.config import Config, RendererSettings, parse_config
from scene_recon.data.synthetic import SyntheticSpheresDataset
from scene_recon.models.base import ParameterGroup, ReconstructionModel, weights_sha256
from scene_recon.models.contract import ModelContractError, check_model_contract
from scene_recon.models.normalization import InputNormalization
from scene_recon.registry import MODELS, ConfigError
from scene_recon.rendering import VolumeRenderer
from scene_recon.provenance import evaluation_status
from scene_recon.stages import check_run_can_start, run_evaluation
from scene_recon.structures import FieldOutput, SceneViews
from scene_recon.training import Trainer

from test_rendering import _OracleSphereModel


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


def test_input_normalization_applies_the_named_statistics():
    images = torch.full((1, 3, 2, 2), 0.5)

    assert torch.allclose(InputNormalization("none")(images), images)
    assert torch.allclose(InputNormalization("minus_one_to_one")(images), torch.zeros_like(images))
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    imagenet = InputNormalization("imagenet")(images)
    assert torch.allclose(imagenet[0, :, 0, 0], (0.5 - mean) / std)


def test_input_normalization_is_not_stored_in_checkpoints():
    assert InputNormalization("imagenet").state_dict() == {}


def test_unknown_input_normalization_is_a_config_error():
    with pytest.raises(ConfigError, match="Available"):
        InputNormalization("cifar")


class _RayAwareModel(_MinimalModel):
    """Relies on what the pipeline promises: each ray's samples lie on that ray, ordered near to far."""

    def query_field(self, scene_encoding, points, view_directions, level):
        steps = points[:, :, 1:] - points[:, :, :-1]
        along_ray = (steps * view_directions[:, :, None]).sum(dim=-1)
        across_ray = steps - along_ray[..., None] * view_directions[:, :, None]
        if (along_ray < 0).any():
            raise ValueError("samples are not ordered from near to far")
        if across_ray.abs().max() > 1e-4:
            raise ValueError("samples are not on the given ray")
        return super().query_field(scene_encoding, points, view_directions, level)


def test_a_model_relying_on_ray_order_passes_the_check_and_renders():
    model = _RayAwareModel()

    check_model_contract(model)
    renderer = VolumeRenderer(
        RendererSettings(
            num_coarse_samples=8,
            num_fine_samples=8,
            num_fine_depth_samples=4,
            fine_depth_std=0.01,
            background_color=[1.0, 1.0, 1.0],
            ray_chunk_size=50,
        )
    )
    views = SyntheticSpheresDataset(num_scenes=1, num_views=2, image_size=8, seed=0).load_scene(0)
    renderer.render_images(model, None, views)  # the real rendering path must accept it too


def test_the_check_uses_the_configured_numbers_of_samples():
    seen = {}

    class RecordingModel(_MinimalModel):
        def query_field(self, scene_encoding, points, view_directions, level):
            seen[level] = points.shape[2]
            return super().query_field(scene_encoding, points, view_directions, level)

    check_model_contract(RecordingModel(), num_coarse_samples=5, num_fine_samples=3)

    assert seen == {"coarse": 5, "fine": 8}


def test_parameters_that_never_receive_a_gradient_are_reported():
    class UnusedLayer(_MinimalModel):
        def __init__(self) -> None:
            super().__init__()
            self.unused = nn.Linear(2, 2)

    with pytest.raises(ModelContractError, match="receive no gradient.*unused"):
        check_model_contract(UnusedLayer())


def test_non_finite_gradients_are_reported():
    class InfiniteSlope(_MinimalModel):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))

        def query_field(self, scene_encoding, points, view_directions, level):
            field = super().query_field(scene_encoding, points, view_directions, level)
            # Zero in value, but the square root's slope at zero is infinite.
            hidden_nan = 0 * torch.sqrt(self.scale - self.scale.detach())
            return FieldOutput(rgb=field.rgb, density=field.density + hidden_nan)

    with pytest.raises(ModelContractError, match="NaN or infinite gradients"):
        check_model_contract(InfiniteSlope())


def test_the_check_leaves_no_gradients_and_restores_the_mode():
    model = _MinimalModel()
    model.eval()

    check_model_contract(model)

    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())


# ---------------------------------------------------------------- stage-aware checks (R6)


class _FrozenModel(_MinimalModel):
    """A model used only for inference: nothing in it is trainable."""

    def __init__(self) -> None:
        super().__init__()
        self.requires_grad_(False)


if "frozen_test_model" not in MODELS.names():
    MODELS.register("frozen_test_model")(_FrozenModel)


class _TwoViewFusion(_MinimalModel):
    """Has a learned fusion weight that only the two-view path uses."""

    def __init__(self) -> None:
        super().__init__()
        self.two_view_weight = nn.Parameter(torch.ones(4))

    def encode_scene(self, source_views: SceneViews):
        return source_views.num_views

    def query_field(self, scene_encoding, points, view_directions, level):
        output = self.layer(points)
        if scene_encoding == 2:
            output = output * self.two_view_weight
        return FieldOutput(rgb=torch.sigmoid(output[..., :3]), density=torch.relu(output[..., 3]))


def test_a_frozen_model_passes_inference_checks_without_a_backward_pass():
    check_model_contract(_FrozenModel(), inference_view_counts=[1, 2], training_view_counts=None)


def test_a_frozen_model_is_told_clearly_that_it_cannot_be_trained():
    with pytest.raises(ModelContractError, match="no trainable parameters"):
        check_model_contract(_FrozenModel(), training_view_counts=[1, 2])


def test_preflight_of_a_frozen_model_depends_on_the_requested_stages(smoke_config: Config):
    config = parse_config({**smoke_config.raw, "model": {"name": "frozen_test_model"}})

    check_run_can_start(config, ["evaluate", "profile"])
    with pytest.raises(ModelContractError, match="no trainable parameters"):
        check_run_can_start(config, ["train"])


def test_a_parameter_used_only_with_two_views_passes_when_two_view_training_is_configured():
    check_model_contract(_TwoViewFusion(), training_view_counts=[1, 2, 3])

    with pytest.raises(ModelContractError, match=r"training view counts \[1, 3\].*two_view_weight"):
        check_model_contract(_TwoViewFusion(), training_view_counts=[1, 3])


# ---------------------------------------------------------------- models without parameters (#3 N2)


class _AnalyticSphere(_OracleSphereModel):
    """The rendering tests' analytic model, with defaults so a config can select it. It has no parameters."""

    def __init__(self, radius: float = 0.5) -> None:
        super().__init__(radius, torch.tensor([0.4, 0.5, 0.6]))


class _BufferOnlyModel(ReconstructionModel):
    """Keeps its fixed values in a buffer, not in parameters. They are random, drawn when it is built."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weights", torch.rand(3, 4))

    def encode_scene(self, source_views: SceneViews):
        return None

    def query_field(self, scene_encoding, points, view_directions, level):
        output = points @ self.weights
        return FieldOutput(rgb=torch.sigmoid(output[..., :3]), density=torch.relu(output[..., 3]))


if "analytic_sphere_test_model" not in MODELS.names():
    MODELS.register("analytic_sphere_test_model")(_AnalyticSphere)
if "random_buffer_test_model" not in MODELS.names():
    MODELS.register("random_buffer_test_model")(_BufferOnlyModel)


def test_a_parameterless_model_passes_the_inference_check():
    """The review's direct reproduction."""
    model = _OracleSphereModel(0.5, torch.tensor([0.4, 0.5, 0.6]))
    assert list(model.parameters()) == []

    check_model_contract(model, inference_view_counts=[1], training_view_counts=None)


def test_a_buffer_only_model_passes_on_its_device():
    model = _BufferOnlyModel()

    check_model_contract(model, training_view_counts=None)  # device taken from the buffer
    check_model_contract(model, training_view_counts=None, device=torch.device("cpu"))


def test_a_parameterless_model_passes_inference_preflight_but_cannot_be_trained(smoke_config: Config):
    config = parse_config({**smoke_config.raw, "model": {"name": "analytic_sphere_test_model"}})

    check_run_can_start(config, ["evaluate", "profile"])
    with pytest.raises(ModelContractError, match="no trainable parameters"):
        check_run_can_start(config, ["train"])


def test_a_model_without_trainable_parameters_is_evaluated_as_built(smoke_config: Config):
    config = parse_config({**smoke_config.raw, "model": {"name": "analytic_sphere_test_model"}})

    report = run_evaluation(config)

    assert report["checkpoint"] is None
    assert report["untrained_model"]["config"] == {"name": "analytic_sphere_test_model"}
    assert len(report["untrained_model"]["weights_sha256"]) == 64
    assert evaluation_status(config).is_current
    changed = parse_config({**config.raw, "model": {"name": "analytic_sphere_test_model", "radius": 0.3}})
    assert evaluation_status(changed).state == "stale"  # other settings, so another identity


def test_a_trainable_model_without_a_checkpoint_is_not_evaluated(smoke_config: Config):
    with pytest.raises(ConfigError, match="no checkpoint to evaluate"):
        check_run_can_start(smoke_config, ["evaluate"])
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        run_evaluation(smoke_config)


def test_a_model_with_random_fixed_values_is_evaluated_reproducibly(smoke_config: Config):
    """The caveat of the validation comment on #3: random buffers, drawn whenever the model is built."""
    config = parse_config({**smoke_config.raw, "model": {"name": "random_buffer_test_model"}})

    first = run_evaluation(config)

    assert evaluation_status(config).is_current  # rebuilt for the check with the same random values
    assert run_evaluation(config)["protocols"] == first["protocols"]  # the same model, so the same scores
    other_seed = parse_config({**config.raw, "experiment": {**config.raw["experiment"], "seed": 99}})
    status = evaluation_status(other_seed)
    assert status.state == "stale" and "other settings or weights" in status.reason


def test_a_seeded_model_starts_from_the_weights_training_starts_from(smoke_config: Config):
    assert weights_sha256(build_seeded_model(smoke_config)) == weights_sha256(Trainer(smoke_config).model)


def test_building_a_seeded_model_leaves_the_global_random_state_alone(smoke_config: Config):
    def draw() -> tuple[float, float, float]:
        return torch.rand(1).item(), random.random(), float(np.random.rand())

    def seed() -> None:
        torch.manual_seed(5)
        random.seed(5)
        np.random.seed(5)

    seed()
    expected = draw()
    seed()
    build_seeded_model(smoke_config)

    assert draw() == expected
