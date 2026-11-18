"""A self-check for new models.

``check_model_contract`` runs a model on small random scenes and verifies
everything the pipeline relies on:

* output shapes and value ranges, for 1 and 3 source views, at both
  rendering levels, with the run's own numbers of samples per ray;
* the same in evaluation mode;
* every trainable parameter receives a finite gradient, so nothing is dead
  weight and nothing explodes;
* a consistent set of parameter groups.

The 3D points it queries are real samples along real camera rays, ordered
from near to far, exactly as the renderer produces them, so a model that
relies on that ordering (e.g. attention along each ray) is checked on the
input it will actually see.

Run it from the command line with ``scene-recon check-model`` before starting
a long training job; a violation found here costs seconds instead of a
failed run. Suites run it for every run before starting.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from scene_recon.geometry import look_at_pose, rays_through_pixels
from scene_recon.models.base import LEVEL_COARSE, LEVEL_FINE, ReconstructionModel
from scene_recon.rendering.sampling import sample_stratified_depths
from scene_recon.structures import Cameras, FieldOutput, RayBundle, SceneViews

# Size of the random test problem. Small enough to run instantly on a CPU.
_NUM_SCENES = 2
_NUM_RAYS = 5
_SOURCE_VIEW_COUNTS = (1, 3)
_NEAR, _FAR = 0.8, 1.8
_CAMERA_DISTANCE = 1.3


class ModelContractError(AssertionError):
    """Raised when a model does not behave as the pipeline requires."""


def check_model_contract(
    model: ReconstructionModel,
    image_size: int | tuple[int, int] = 64,
    num_coarse_samples: int = 7,
    num_fine_samples: int = 7,
) -> None:
    """Verify that a model satisfies the ``ReconstructionModel`` contract.

    The model is run on whatever device it lives on, and left with its
    original training/evaluation mode and without gradients.

    Args:
        model: The model to check.
        image_size: Size of the random test images, as ``(height, width)``
            or a single number for square images. Use the resolution of the
            dataset if the encoder depends on it (e.g. fixed position embeddings).
        num_coarse_samples: Samples per ray in the coarse pass, as configured
            in ``renderer.num_coarse_samples``.
        num_fine_samples: Extra samples per ray in the fine pass, as
            configured in ``renderer.num_fine_samples``. 0 skips the fine level.

    Raises:
        ModelContractError: With a description of the first violation found.
    """
    if not isinstance(model, ReconstructionModel):
        raise ModelContractError(f"{type(model).__name__} must subclass ReconstructionModel.")
    check_parameter_groups(model)

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    device = next(model.parameters()).device
    sample_counts = {LEVEL_COARSE: num_coarse_samples}
    if num_fine_samples > 0:
        sample_counts[LEVEL_FINE] = num_coarse_samples + num_fine_samples

    was_training = model.training
    model.zero_grad(set_to_none=True)
    try:
        for num_source_views in _SOURCE_VIEW_COUNTS:
            source_views = _random_source_views(num_source_views, image_size).to(device)
            rays = _random_rays(image_size, device)

            # As in training: outputs are checked and every level contributes to a backward pass.
            model.train()
            total = 0.0
            for level, num_samples in sample_counts.items():
                where = f"query_field(level='{level}') with {num_source_views} source view(s), training mode"
                field = _query(model, source_views, rays, num_samples, level)
                _check_field_output(field, num_samples, where)
                total = total + field.rgb.mean() + field.density.mean()
            total.backward()

            # As in evaluation.
            model.eval()
            with torch.no_grad():
                for level, num_samples in sample_counts.items():
                    where = f"query_field(level='{level}') with {num_source_views} source view(s), evaluation mode"
                    _check_field_output(_query(model, source_views, rays, num_samples, level), num_samples, where)

        _check_gradients(model)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def check_parameter_groups(model: ReconstructionModel) -> None:
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


def _query(
    model: ReconstructionModel, source_views: SceneViews, rays: RayBundle, num_samples: int, level: str
) -> FieldOutput:
    """Query the model at ``num_samples`` ordered samples along each ray, like the renderer does."""
    depths = sample_stratified_depths(rays, num_samples, generator=torch.Generator().manual_seed(num_samples))
    points = rays.origins[:, :, None] + depths[..., None] * rays.directions[:, :, None]
    scene_encoding = model.encode_scene(source_views)
    return model.query_field(scene_encoding, points, rays.directions, level)


def _check_field_output(field: FieldOutput, num_samples: int, where: str) -> None:
    if not isinstance(field, FieldOutput):
        raise ModelContractError(f"{where} must return a FieldOutput, got {type(field).__name__}.")

    _check_tensor(field.rgb, "rgb", (_NUM_SCENES, _NUM_RAYS, num_samples, 3), where)
    _check_tensor(field.density, "density", (_NUM_SCENES, _NUM_RAYS, num_samples), where)

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


def _check_gradients(model: ReconstructionModel) -> None:
    """Every trainable parameter must receive a finite gradient from the outputs."""
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    unused = [name for name, parameter in trainable if parameter.grad is None]
    if unused:
        raise ModelContractError(
            f"{len(unused)} trainable parameter tensor(s) receive no gradient, so training would never change "
            f"them (e.g. {', '.join(unused[:3])}). Remove them, or freeze them with requires_grad = False."
        )
    non_finite = [name for name, parameter in trainable if not torch.isfinite(parameter.grad).all()]
    if non_finite:
        raise ModelContractError(
            f"{len(non_finite)} parameter tensor(s) receive NaN or infinite gradients (e.g. {', '.join(non_finite[:3])})."
        )


def _random_source_views(num_views: int, image_size: tuple[int, int]) -> SceneViews:
    """Random images with valid cameras placed on a circle around the origin, looking inwards."""
    world_up = torch.tensor([0.0, 0.0, 1.0])
    poses = []
    for view in range(num_views):
        angle = 2 * math.pi * view / num_views
        position = torch.tensor([_CAMERA_DISTANCE * math.cos(angle), _CAMERA_DISTANCE * math.sin(angle), 0.5])
        poses.append(look_at_pose(position, torch.zeros(3), world_up))
    camera_to_world = torch.stack(poses).expand(_NUM_SCENES, num_views, 4, 4).clone()

    return SceneViews(
        images=torch.rand(_NUM_SCENES, num_views, 3, *image_size),
        cameras=Cameras(camera_to_world, _intrinsics(image_size).expand(_NUM_SCENES, num_views, 3, 3).clone()),
        near=torch.full((_NUM_SCENES,), _NEAR),
        far=torch.full((_NUM_SCENES,), _FAR),
        scene_ids=[f"random_{index}" for index in range(_NUM_SCENES)],
    )


def _random_rays(image_size: tuple[int, int], device: torch.device) -> RayBundle:
    """Rays through random pixels of a target camera that is not one of the source cameras."""
    height, width = image_size
    pose = look_at_pose(torch.tensor([0.9, -0.9, 0.6]), torch.zeros(3), torch.tensor([0.0, 0.0, 1.0]))
    cameras = Cameras(
        camera_to_world=pose.expand(_NUM_SCENES, _NUM_RAYS, 4, 4),
        intrinsics=_intrinsics(image_size).expand(_NUM_SCENES, _NUM_RAYS, 3, 3),
    )
    generator = torch.Generator().manual_seed(0)
    pixels = torch.rand(_NUM_SCENES, _NUM_RAYS, 2, generator=generator) * torch.tensor([width, height])
    origins, directions = rays_through_pixels(cameras, pixels)
    return RayBundle(
        origins=origins.to(device),
        directions=directions.to(device),
        near=torch.full((_NUM_SCENES,), _NEAR, device=device),
        far=torch.full((_NUM_SCENES,), _FAR, device=device),
    )


def _intrinsics(image_size: tuple[int, int]) -> Tensor:
    height, width = image_size
    focal = 1.2 * width
    return torch.tensor([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
