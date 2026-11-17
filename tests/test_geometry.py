import torch

from scene_recon.geometry import (
    look_at_pose,
    pixel_centers,
    project_to_pixels,
    rays_through_pixels,
    rotate_directions_to_camera,
    sample_features_at_pixels,
    world_to_camera,
)
from scene_recon.structures import Cameras

IMAGE_SIZE = 8


def _camera() -> Cameras:
    """A camera placed off-axis, looking at the origin."""
    pose = look_at_pose(torch.tensor([1.0, -0.7, 0.5]), torch.zeros(3), torch.tensor([0.0, 0.0, 1.0]))
    intrinsics = torch.tensor([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]])
    return Cameras(pose, intrinsics)


def test_look_at_pose_is_a_rotation_that_faces_the_target():
    position = torch.tensor([1.0, -0.7, 0.5])
    pose = look_at_pose(position, torch.zeros(3), torch.tensor([0.0, 0.0, 1.0]))
    rotation = pose[:3, :3]

    assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-6)
    assert torch.isclose(torch.linalg.det(rotation), torch.tensor(1.0), atol=1e-6)
    forward = rotation[:, 2]
    assert torch.allclose(forward, -position / position.norm(), atol=1e-6)


def test_world_up_points_up_in_the_image():
    # In the OpenCV convention the camera's y axis points down, so a point
    # above the target must project to a smaller y coordinate.
    camera = _camera()
    point_above_origin = torch.tensor([[[0.0, 0.0, 0.2]]])
    points_camera = world_to_camera(point_above_origin, camera.camera_to_world[None, None])
    pixel = project_to_pixels(points_camera, camera.intrinsics[None, None])[0, 0, 0]

    principal_point_y = camera.intrinsics[1, 2]
    assert pixel[1] < principal_point_y


def test_projecting_points_on_a_ray_returns_the_rays_pixel():
    camera = _camera()
    pixels = pixel_centers(IMAGE_SIZE, IMAGE_SIZE)
    per_pixel_camera = Cameras(
        camera.camera_to_world.expand(len(pixels), 4, 4), camera.intrinsics.expand(len(pixels), 3, 3)
    )
    origins, directions = rays_through_pixels(per_pixel_camera, pixels)
    points_world = origins + 1.7 * directions

    points_camera = world_to_camera(points_world[None], camera.camera_to_world[None, None])
    reprojected = project_to_pixels(points_camera, camera.intrinsics[None, None])[0, 0]

    assert torch.allclose(reprojected, pixels, atol=1e-4)
    assert torch.allclose(directions.norm(dim=-1), torch.ones(len(pixels)), atol=1e-6)


def test_directions_rotate_like_points_without_translation():
    camera = _camera()
    directions_world = torch.nn.functional.normalize(torch.randn(1, 5, 3), dim=-1)
    pose = camera.camera_to_world[None, None]

    rotated = rotate_directions_to_camera(directions_world, pose)
    origin_camera = world_to_camera(torch.zeros(1, 5, 3), pose)
    expected = world_to_camera(directions_world, pose) - origin_camera

    assert torch.allclose(rotated, expected, atol=1e-6)


def test_feature_lookup_at_pixel_centres_returns_the_exact_values():
    feature_maps = torch.rand(1, 1, 3, IMAGE_SIZE, IMAGE_SIZE)
    pixels = pixel_centers(IMAGE_SIZE, IMAGE_SIZE)[None, None]

    sampled = sample_features_at_pixels(feature_maps, pixels, (IMAGE_SIZE, IMAGE_SIZE))

    expected = feature_maps[0, 0].permute(1, 2, 0).reshape(-1, 3)
    assert torch.allclose(sampled[0, 0], expected, atol=1e-6)


def test_feature_lookup_works_on_downsampled_feature_maps():
    # A feature map that stores the x coordinate at half resolution: looking
    # it up at any image pixel must interpolate to that pixel's x coordinate.
    half_size = IMAGE_SIZE // 2
    x_of_feature_centres = (torch.arange(half_size) + 0.5) * 2.0
    feature_maps = x_of_feature_centres.expand(1, 1, 1, half_size, half_size)
    pixels = torch.tensor([[[[3.0, 2.0], [4.5, 6.0]]]])

    sampled = sample_features_at_pixels(feature_maps, pixels, (IMAGE_SIZE, IMAGE_SIZE))

    assert torch.allclose(sampled[0, 0, :, 0], torch.tensor([3.0, 4.5]), atol=1e-5)
