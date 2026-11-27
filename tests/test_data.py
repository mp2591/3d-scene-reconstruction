from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scene_recon.data.srn import SRNDataset
from scene_recon.geometry import rays_through_pixels
from scene_recon.structures import Cameras
from scene_recon.data.synthetic import SyntheticSpheresDataset
from scene_recon.registry import ConfigError

from conftest import write_srn_scene


class TestSRNDataset:
    def test_reads_images_poses_and_intrinsics(self, tmp_path: Path):
        write_srn_scene(tmp_path / "scene_a", num_views=4, image_size=16, calibrated_size=16)
        write_srn_scene(tmp_path / "scene_b", num_views=4, image_size=16, calibrated_size=16)
        dataset = SRNDataset(str(tmp_path), near=0.8, far=1.8, pixel_centers="half_integer")

        scene = dataset.load_scene(1, view_indices=[2, 0])

        assert len(dataset) == 2
        assert dataset.num_views(1) == 4
        assert scene.scene_ids == ["scene_b"]
        assert scene.images.shape == (1, 2, 3, 16, 16)
        # Views come back in the requested order.
        assert torch.allclose(scene.images[0, :, 0, 0, 0], torch.tensor([20.0, 0.0]) / 255)
        assert torch.allclose(scene.cameras.camera_to_world[0, :, 0, 3], torch.tensor([2.0, 0.0]))
        assert torch.allclose(scene.cameras.intrinsics[0, 0, 0], torch.tensor([131.25, 0.0, 8.0]))
        assert scene.near.item() == pytest.approx(0.8) and scene.far.item() == pytest.approx(1.8)

    def test_intrinsics_follow_the_resolution_of_the_images(self, tmp_path: Path):
        # Images stored at half the resolution the intrinsics were calibrated for.
        write_srn_scene(tmp_path / "scene_a", num_views=1, image_size=8, calibrated_size=16)

        scene = SRNDataset(str(tmp_path), near=0.8, far=1.8, pixel_centers="half_integer").load_scene(0)

        assert torch.allclose(scene.cameras.intrinsics[0, 0, 0], torch.tensor([131.25 / 2, 0.0, 4.0]))

    def test_integer_pixel_centres_shift_the_principal_point_by_half_a_pixel(self, tmp_path: Path):
        # Calibrated at 16x16, stored at 8x8: the shift is applied before scaling.
        write_srn_scene(tmp_path / "scene_a", num_views=1, image_size=8, calibrated_size=16)

        scene = SRNDataset(str(tmp_path), near=0.8, far=1.8, pixel_centers="integer").load_scene(0)

        assert torch.allclose(scene.cameras.intrinsics[0, 0, 0, 2], torch.tensor((8.0 + 0.5) / 2))

    @pytest.mark.parametrize(
        ("pixel_centers", "expected_direction"),
        [("integer", [0.0, 0.0, 1.0]), ("half_integer", [0.00380947, 0.00380947, 0.99998546])],
    )
    def test_ray_through_the_central_pixel_matches_the_review_example(
        self, tmp_path: Path, pixel_centers: str, expected_direction: list[float]
    ):
        """f = 131.25, cx = cy = 64 at 128x128 (the SRN Cars intrinsics), camera at the origin.

        The original SRN and pixelNeRF code casts the ray through pixel index
        (64, 64) straight along the optical axis; the "integer" setting
        reproduces that, "half_integer" is half a pixel off.
        """
        write_srn_scene(tmp_path / "scene_a", num_views=1, image_size=128, calibrated_size=128)
        scene = SRNDataset(str(tmp_path), near=0.8, far=1.8, pixel_centers=pixel_centers).load_scene(0)
        camera = Cameras(scene.cameras.camera_to_world[0, 0], scene.cameras.intrinsics[0, 0])
        camera = Cameras(torch.eye(4), camera.intrinsics)  # look along +z from the origin

        _, direction = rays_through_pixels(camera, torch.tensor([64.0, 64.0]) + 0.5)  # pixel (64, 64)

        assert torch.allclose(direction, torch.tensor(expected_direction), atol=1e-6)

    def test_unknown_pixel_convention_is_a_config_error(self, tmp_path: Path):
        write_srn_scene(tmp_path / "scene_a", num_views=1, image_size=8, calibrated_size=8)

        with pytest.raises(ConfigError, match="pixel_centers"):
            SRNDataset(str(tmp_path), near=0.8, far=1.8, pixel_centers="corner")

    def test_integer_convention_on_resized_images_keeps_continuous_scaling(self, tmp_path: Path):
        """Calibrated at 128 (f = 131.25, cx = cy = 64), stored at 64x64: the review example.

        The loader scales the shifted principal point, (64 + 0.5) * 0.5 = 32.25.
        pixelNeRF would use 64 * 0.5 + 0.5 = 32.5, so the ray through pixel
        (32, 32) is a quarter pixel off the optical axis here, as documented.
        """
        write_srn_scene(tmp_path / "scene_a", num_views=1, image_size=64, calibrated_size=128)
        scene = SRNDataset(str(tmp_path), near=0.8, far=1.8, pixel_centers="integer").load_scene(0)
        camera = Cameras(torch.eye(4), scene.cameras.intrinsics[0, 0])

        _, direction = rays_through_pixels(camera, torch.tensor([32.0, 32.0]) + 0.5)

        assert scene.cameras.intrinsics[0, 0, 0, 2].item() == pytest.approx(32.25)
        assert torch.allclose(direction, torch.tensor([0.00380947, 0.00380947, 0.99998558]), atol=1e-6)


    def test_missing_directory_is_a_config_error(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="does not exist"):
            SRNDataset(str(tmp_path / "nowhere"), near=0.8, far=1.8, pixel_centers="half_integer")


class TestSyntheticSpheres:
    def test_scenes_are_deterministic_and_independent_of_the_requested_views(self):
        dataset = SyntheticSpheresDataset(num_scenes=2, num_views=5, image_size=16, seed=3)

        all_views = dataset.load_scene(1)
        some_views = dataset.load_scene(1, view_indices=[4, 1])

        assert torch.equal(some_views.images[0, 0], all_views.images[0, 4])
        assert torch.equal(some_views.cameras.camera_to_world[0, 1], all_views.cameras.camera_to_world[0, 1])

    def test_images_show_an_object_on_a_white_background(self):
        scene = SyntheticSpheresDataset(num_scenes=1, num_views=2, image_size=32, seed=0).load_scene(0)

        corner_pixel = scene.images[0, 0, :, 0, 0]
        centre_pixel = scene.images[0, 0, :, 16, 16]
        assert torch.equal(corner_pixel, torch.ones(3))
        assert not torch.equal(centre_pixel, torch.ones(3))
        assert scene.images.min() >= 0 and scene.images.max() <= 1
