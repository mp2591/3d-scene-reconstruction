"""A self-check for new models.

``check_model_contract`` runs a model on small random inputs and verifies
everything the pipeline relies on: output shapes and value ranges, support
for a varying number of source views, and a consistent set of parameter
groups. Run it from the command line with ``scene-recon check-model`` before
starting a long training job; a violation found here costs seconds instead of
a failed run.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from scene_recon.geometry import look_at_pose
from scene_recon.models.base import LEVEL_COARSE, LEVEL_FINE, ReconstructionModel
from scene_recon.structures import Cameras, FieldOutput, SceneViews

# Sizes of the random test problem. Small enough to run instantly on a CPU.
_NUM_SCENES = 2
_NUM_RAYS = 5
_NUM_SAMPLES = 7
_SOURCE_VIEW_COUNTS = (1, 3)


class ModelContractError(AssertionError):
    """Raised when a model does not behave as the pipeline requires."""


def check_model_contract(model: ReconstructionModel, image_size: int | tuple[int, int] = 64) -> None:
    """Verify that a model satisfies the ``ReconstructionModel`` contract.

    Args:
        model: The model to check. It is run on whatever device it lives on.
        image_size: Size of the random test images, as ``(height, width)``
            or a single number for square images. Use the resolution of the
            dataset if the encoder depends on it (e.g. fixed position embeddings).

    Raises:
        ModelContractError: With a description of the first violation found.
    """
    if not isinstance(model, ReconstructionModel):
        raise ModelContractError(f"{type(model).__name__} must subclass ReconstructionModel.")

    _check_parameter_groups(model)

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    device = next(model.parameters()).device
    for num_source_views in _SOURCE_VIEW_COUNTS:
        source_views = _random_source_views(num_source_views, image_size).to(device)
        points = (torch.rand(_NUM_SCENES, _NUM_RAYS, _NUM_SAMPLES, 3, device=device) - 0.5) * 0.8
        view_directions = torch.nn.functional.normalize(
            torch.randn(_NUM_SCENES, _NUM_RAYS, 3, device=device), dim=-1
        )

        scene_encoding = model.encode_scene(source_views)
        for level in (LEVEL_COARSE, LEVEL_FINE):
            where = f"query_field(level='{level}') with {num_source_views} source view(s)"
            field = model.query_field(scene_encoding, points, view_directions, level)
            _check_field_output(field, where)


def _check_parameter_groups(model: ReconstructionModel) -> None:
    """Every trainable parameter must belong to exactly one, uniquely named group."""
    groups = model.parameter_groups()

    names = [group.name for group in groups]
    if len(set(names)) != len(names):
        raise ModelContractError(f"parameter_groups() returned duplicate group names: {names}.")

    times_seen = {id(parameter): 0 for parameter in model.parameters() if parameter.requires_grad}
    for group in groups:
        for parameter in group.parameters:
            if id(parameter) in times_seen:
                times_seen[id(parameter)] += 1

    num_missing = sum(count == 0 for count in times_seen.values())
    num_repeated = sum(count > 1 for count in times_seen.values())
    if num_missing:
        raise ModelContractError(
            f"{num_missing} trainable parameter tensor(s) are in no parameter group and would never be trained."
        )
    if num_repeated:
        raise ModelContractError(f"{num_repeated} parameter tensor(s) appear in more than one parameter group.")


def _check_field_output(field: FieldOutput, where: str) -> None:
    if not isinstance(field, FieldOutput):
        raise ModelContractError(f"{where} must return a FieldOutput, got {type(field).__name__}.")

    expected_rgb_shape = (_NUM_SCENES, _NUM_RAYS, _NUM_SAMPLES, 3)
    expected_density_shape = (_NUM_SCENES, _NUM_RAYS, _NUM_SAMPLES)
    _check_tensor(field.rgb, "rgb", expected_rgb_shape, where)
    _check_tensor(field.density, "density", expected_density_shape, where)

    if field.rgb.min() < 0 or field.rgb.max() > 1:
        raise ModelContractError(
            f"{where}: rgb must lie in [0, 1] (e.g. apply a sigmoid), "
            f"got values in [{field.rgb.min():.3f}, {field.rgb.max():.3f}]."
        )
    if field.density.min() < 0:
        raise ModelContractError(
            f"{where}: density must be non-negative (e.g. apply a ReLU or softplus), "
            f"got a minimum of {field.density.min():.3f}."
        )


def _check_tensor(tensor: Tensor, name: str, expected_shape: tuple[int, ...], where: str) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise ModelContractError(f"{where}: {name} has shape {tuple(tensor.shape)}, expected {expected_shape}.")
    if not torch.isfinite(tensor).all():
        raise ModelContractError(f"{where}: {name} contains NaN or infinite values.")


def _random_source_views(num_views: int, image_size: tuple[int, int]) -> SceneViews:
    """Random images with valid cameras placed on a circle around the origin, looking inwards."""
    world_up = torch.tensor([0.0, 0.0, 1.0])
    poses = []
    for view in range(num_views):
        angle = 2 * math.pi * view / num_views
        position = torch.tensor([1.3 * math.cos(angle), 1.3 * math.sin(angle), 0.5])
        poses.append(look_at_pose(position, torch.zeros(3), world_up))
    camera_to_world = torch.stack(poses).expand(_NUM_SCENES, num_views, 4, 4).clone()

    height, width = image_size
    focal = 1.2 * width
    intrinsics = torch.tensor([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    intrinsics = intrinsics.expand(_NUM_SCENES, num_views, 3, 3).clone()

    return SceneViews(
        images=torch.rand(_NUM_SCENES, num_views, 3, height, width),
        cameras=Cameras(camera_to_world, intrinsics),
        near=torch.full((_NUM_SCENES,), 0.8),
        far=torch.full((_NUM_SCENES,), 1.8),
        scene_ids=[f"random_{index}" for index in range(_NUM_SCENES)],
    )
