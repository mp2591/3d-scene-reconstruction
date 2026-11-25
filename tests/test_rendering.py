import dataclasses
import math
from typing import Any

import pytest
import torch
from torch import Tensor

from scene_recon.config import RendererSettings
from scene_recon.data.synthetic import SyntheticSpheresDataset
from scene_recon.build import build_model
from scene_recon.config import Config
from scene_recon.metrics import PSNR
from scene_recon.models.base import ReconstructionModel
from scene_recon.rendering import VolumeRenderer, rays_for_full_images, sample_training_rays
from scene_recon.rendering.sampling import sample_importance_depths, sample_stratified_depths
from scene_recon.rendering.volume import composite_along_rays
from scene_recon.structures import FieldOutput, RayBundle, SceneViews

WHITE = torch.ones(3)


def _rays(num_scenes: int = 2, num_rays: int = 5) -> RayBundle:
    directions = torch.nn.functional.normalize(torch.randn(num_scenes, num_rays, 3), dim=-1)
    return RayBundle(
        origins=torch.zeros(num_scenes, num_rays, 3),
        directions=directions,
        near=torch.full((num_scenes,), 0.8),
        far=torch.full((num_scenes,), 1.8),
    )


def _settings(num_coarse_samples: int, num_fine_samples: int) -> RendererSettings:
    return RendererSettings(
        num_coarse_samples=num_coarse_samples,
        num_fine_samples=num_fine_samples,
        background_color=[1.0, 1.0, 1.0],
        ray_chunk_size=500,
    )


class TestStratifiedSampling:
    @pytest.mark.parametrize("randomize", [False, True])
    def test_depths_are_sorted_and_within_near_and_far(self, randomize: bool):
        generator = torch.Generator().manual_seed(0) if randomize else None
        depths = sample_stratified_depths(_rays(), num_samples=16, generator=generator)

        assert depths.shape == (2, 5, 16)
        assert (depths[..., 1:] >= depths[..., :-1]).all()
        assert depths.min() >= 0.8 and depths.max() <= 1.8

    def test_without_randomisation_samples_sit_at_bin_centres(self):
        depths = sample_stratified_depths(_rays(), num_samples=4, generator=None)

        expected = torch.tensor([0.925, 1.175, 1.425, 1.675])
        assert torch.allclose(depths[0, 0], expected, atol=1e-6)


class TestImportanceSampling:
    def test_new_samples_concentrate_where_the_weights_are(self):
        depths = torch.linspace(0.8, 1.8, 11).expand(1, 1, 11)
        weights = torch.zeros(1, 1, 11)
        weights[..., 5] = 1.0  # all content at depth 1.3

        new_depths = sample_importance_depths(depths, weights, num_samples=64, generator=torch.Generator())

        assert new_depths.shape == (1, 1, 64)
        assert (new_depths[..., 1:] >= new_depths[..., :-1]).all()
        # The weight of sample 5 belongs to the bin between its neighbouring midpoints.
        assert ((new_depths > 1.25) & (new_depths < 1.35)).float().mean() > 0.95

    def test_rays_that_hit_nothing_still_get_valid_samples(self):
        depths = torch.linspace(0.8, 1.8, 11).expand(1, 3, 11)
        weights = torch.zeros(1, 3, 11)

        new_depths = sample_importance_depths(depths, weights, num_samples=8, generator=None)

        assert torch.isfinite(new_depths).all()
        assert new_depths.min() >= 0.8 and new_depths.max() <= 1.8


class TestCompositing:
    def test_empty_space_shows_the_background(self):
        depths = torch.linspace(0.8, 1.8, 8).expand(1, 1, 8)
        field = FieldOutput(rgb=torch.rand(1, 1, 8, 3), density=torch.zeros(1, 1, 8))

        output = composite_along_rays(field, depths, WHITE)

        assert torch.allclose(output.rgb, torch.ones(1, 1, 3))
        assert torch.allclose(output.opacity, torch.zeros(1, 1))

    def test_opaque_surface_hides_everything_behind_it(self):
        depths = torch.linspace(0.8, 1.8, 11).expand(1, 1, 11)
        density = torch.zeros(1, 1, 11)
        density[..., 4:] = 1e4  # solid from the fifth sample (depth 1.2) onwards
        rgb = torch.zeros(1, 1, 11, 3)
        rgb[..., 4, :] = torch.tensor([1.0, 0.0, 0.0])  # the surface is red, the interior black

        output = composite_along_rays(FieldOutput(rgb, density), depths, WHITE)

        assert torch.allclose(output.rgb, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=1e-4)
        assert torch.allclose(output.depth, torch.tensor([[1.2]]), atol=1e-4)
        assert torch.allclose(output.opacity, torch.ones(1, 1), atol=1e-4)

    def test_weights_never_sum_to_more_than_one(self):
        depths = torch.linspace(0.8, 1.8, 16).expand(2, 3, 16)
        field = FieldOutput(rgb=torch.rand(2, 3, 16, 3), density=torch.rand(2, 3, 16) * 20)

        output = composite_along_rays(field, depths, WHITE)

        assert (output.opacity <= 1.0 + 1e-5).all()


class TestRaySelection:
    def test_training_rays_carry_the_colour_of_their_pixel(self):
        # Rendering the analytic sphere along a sampled ray must reproduce the
        # sampled colour; here we only check the cheaper necessary condition
        # that the colours come from the right images.
        views = SyntheticSpheresDataset(num_scenes=1, num_views=3, image_size=8, seed=0).load_scene(0)
        views.images[:, 0] = 0.1  # make each view a distinct constant colour
        views.images[:, 1] = 0.5
        views.images[:, 2] = 0.9

        rays, target_rgb = sample_training_rays(views, rays_per_scene=200, generator=torch.Generator())

        camera_positions = views.cameras.camera_to_world[0, :, :3, 3]
        view_of_ray = (rays.origins[0][:, None] - camera_positions[None]).norm(dim=-1).argmin(dim=-1)
        expected = torch.tensor([0.1, 0.5, 0.9])[view_of_ray]
        assert torch.allclose(target_rgb[0], expected[:, None].expand(-1, 3))

    def test_full_image_rays_are_ordered_like_image_pixels(self):
        views = SyntheticSpheresDataset(num_scenes=1, num_views=2, image_size=8, seed=0).load_scene(0)

        rays = rays_for_full_images(views)

        assert rays.origins.shape == (1, 2 * 8 * 8, 3)
        # The centre of the image looks straight along the camera's z axis.
        directions = rays.directions.reshape(1, 2, 8, 8, 3)
        central_direction = torch.nn.functional.normalize(directions[0, 0, 3:5, 3:5].mean(dim=(0, 1)), dim=-1)
        camera_forward = views.cameras.camera_to_world[0, 0, :3, 2]
        assert torch.allclose(central_direction, camera_forward, atol=1e-4)


class _OracleSphereModel(ReconstructionModel):
    """Knows the synthetic scene exactly instead of learning it.

    Returns a very high density inside the sphere and the dataset's surface
    colour, so that rendering it must reproduce the dataset's images.
    """

    def __init__(self, radius: float, base_color: Tensor) -> None:
        super().__init__()
        self.radius = radius
        self.base_color = base_color

    def encode_scene(self, source_views: SceneViews) -> None:
        return None

    def query_field(self, scene_encoding: Any, points: Tensor, view_directions: Tensor, level: str) -> FieldOutput:
        distance_from_centre = points.norm(dim=-1)
        normals = points / distance_from_centre[..., None]
        rgb = 0.6 * self.base_color + 0.4 * (0.5 + 0.5 * normals)
        density = (distance_from_centre < self.radius).float() * 1e4
        return FieldOutput(rgb=rgb, density=density)


def test_rendering_the_true_scene_reproduces_the_dataset_images():
    """End-to-end check of ray casting, sampling and compositing against an analytic render."""
    dataset = SyntheticSpheresDataset(num_scenes=1, num_views=3, image_size=24, seed=0)
    views = dataset.load_scene(0)

    radius, base_color, _ = dataset.scene_parameters(0)

    model = _OracleSphereModel(radius, base_color)
    renderer = VolumeRenderer(_settings(num_coarse_samples=64, num_fine_samples=64))
    rendered, depth = renderer.render_images(model, None, views)

    assert rendered.shape == views.images.shape
    assert depth.shape == (1, 3, 24, 24)
    # Only the sphere's silhouette, where a pixel is partly covered, may differ.
    psnr = PSNR()(rendered[0], views.images[0])
    assert psnr.min() > 30


def test_a_partly_transparent_medium_matches_the_exact_integral():
    """A uniform fog of density sigma and colour c fills depths [0.8, 1.3]; behind it is empty space.

    Light passing through fog of thickness L keeps exp(-sigma * L) of its
    intensity, so the ray's opacity is 1 - exp(-sigma * L) and its colour is
    c times that opacity plus the white background times the rest.
    """
    sigma, fog_end, fog_color = 2.0, 1.3, torch.tensor([0.2, 0.4, 0.6])
    depths = torch.linspace(0.8, 1.8, 2001)[:-1].expand(1, 1, 2000)  # 2000 segments of 0.0005
    density = torch.where(depths < fog_end, sigma, 0.0)
    field = FieldOutput(rgb=fog_color.expand(1, 1, 2000, 3), density=density)

    output = composite_along_rays(field, depths, WHITE)

    expected_opacity = 1 - math.exp(-sigma * (fog_end - 0.8))
    assert output.opacity.item() == pytest.approx(expected_opacity, abs=2e-3)
    expected_rgb = fog_color * expected_opacity + WHITE * (1 - expected_opacity)
    assert torch.allclose(output.rgb[0, 0], expected_rgb, atol=2e-3)


def test_rendered_images_do_not_depend_on_the_ray_chunk_size(smoke_config: Config):
    model = build_model(smoke_config).eval()
    views = SyntheticSpheresDataset(num_scenes=1, num_views=3, image_size=12, seed=0).load_scene(0)
    encoding = model.encode_scene(views.select_views([0]))

    small_chunks, _ = VolumeRenderer(dataclasses.replace(_settings(16, 8), ray_chunk_size=7)).render_images(
        model, encoding, views
    )
    one_chunk, _ = VolumeRenderer(dataclasses.replace(_settings(16, 8), ray_chunk_size=100_000)).render_images(
        model, encoding, views
    )

    assert torch.allclose(small_chunks, one_chunk, atol=1e-6)


def test_a_speed_measured_while_the_gpu_memory_overflowed_is_refused(monkeypatch):
    """Review B(c): some drivers move GPU memory to main memory instead of failing, and rendering crawls."""
    from scene_recon import profiling

    renderer = VolumeRenderer(_settings(16, 8))
    gpu = torch.device("cuda:0")
    gb = 10**9

    def check(reserved: int, least_free: int):
        monkeypatch.setattr(profiling, "gpu_memory", lambda device: {"allocated": 3 * gb, "reserved": reserved, "total": 24 * gb})
        return profiling._check_memory_fits(gpu, renderer, least_free, min_free_gpu_memory=0.02)

    with pytest.raises(RuntimeError, match="Lower renderer.ray_chunk_size"):
        check(reserved=25 * gb, least_free=0)  # what the RTX 4090 showed at 8192 rays: reserved beyond the total
    with pytest.raises(RuntimeError, match="Lower renderer.ray_chunk_size"):
        check(reserved=25 * gb, least_free=2 * gb)  # reserved beyond the total, even if some was free at times
    with pytest.raises(RuntimeError, match="close other programs"):
        check(reserved=6 * gb, least_free=gb // 10)  # other programs filled the rest (review F45)

    assert check(reserved=6 * gb, least_free=15 * gb)["least_free"] == 15 * gb
    assert profiling._check_memory_fits(torch.device("cpu"), renderer, None, min_free_gpu_memory=0.02) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 2 * 10**9, reason="needs a CUDA GPU with 2 GB free"
)
def test_the_speed_measurement_reports_gpu_memory_and_refuses_a_full_gpu(smoke_config: Config, monkeypatch):
    """Review F45, F55, F56, F63, F69, F70 and F73: through measure_rendering_speed itself, on a real GPU."""
    from scene_recon import profiling
    from scene_recon.build import build_dataset

    model = build_model(smoke_config).to("cuda")
    views = build_dataset(smoke_config, "test").load_scene(0, [0, 1]).to("cuda")
    renderer = VolumeRenderer(smoke_config.renderer)

    def measure(num_frames: int = 1) -> dict:
        return profiling.measure_rendering_speed(
            model,
            renderer,
            views.select_views([0]),
            views.select_views([1]),
            num_frames=num_frames,
            num_warmup_frames=0,
            min_free_gpu_memory=0.02,
        )

    # Memory PyTorch keeps for reuse after earlier work, such as training in the same process,
    # is released first: the peaks count rendering alone.
    left_over_bytes = 10**9  # far more than rendering the smoke test's small frames reserves
    left_over = torch.empty(left_over_bytes, dtype=torch.uint8, device="cuda")
    del left_over
    speed = measure()
    assert 0 < speed["peak_memory_allocated_gb"] <= speed["peak_memory_reserved_gb"] < left_over_bytes / 1e9
    assert speed["least_free_memory_gb"] > 0

    # Another program filled the GPU during the first frame and released it before the second.
    readings = iter([0, 8 * 10**9, 8 * 10**9])
    monkeypatch.setattr(profiling, "free_gpu_memory", lambda device: next(readings))
    with pytest.raises(RuntimeError, match="at one point only 0.0 GB was free"):
        measure(num_frames=2)

    # The GPU filled up with work that was still queued when the frames had been handed over.
    readings = iter([8 * 10**9, 8 * 10**9, 0])
    model.train()
    frozen = next(model.children())
    frozen.eval()  # a part kept in evaluation mode while the rest trains
    with pytest.raises(RuntimeError, match="at one point only 0.0 GB was free"):
        measure(num_frames=2)
    assert model.training and not frozen.training  # every part's mode restored, although the measurement was refused
