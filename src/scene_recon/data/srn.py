"""Loader for the ShapeNet-SRN benchmark (Sitzmann et al., 2019).

Expected layout of one split directory, e.g. ``cars_train/``::

    <scene_id>/
        intrinsics.txt
        rgb/000000.png, 000001.png, ...
        pose/000000.txt, 000001.txt, ...

``intrinsics.txt`` starts with the line ``focal cx cy <unused>`` and ends
with the line ``height width``. Each pose file holds the 16 numbers of a
row-major 4x4 camera-to-world matrix in the OpenCV convention.

Pixel convention. The pipeline places the centre of pixel (column i, row j)
at image coordinates (i + 0.5, j + 0.5). The original SRN and pixelNeRF code
instead reads these intrinsics with the centre of pixel (i, j) at (i, j),
which differs by half a pixel (about 0.3 degrees for SRN Cars). Which one
matches how the images were rendered can only be settled on real data, so
the choice is an explicit setting, ``pixel_centers``:

* ``"half_integer"``: the intrinsics are used as stored.
* ``"integer"``: the principal point is shifted by +0.5 pixel, so that the
  pipeline casts exactly the rays the original code casts.

To decide, train the same model briefly with each setting and compare
validation quality; the convention that matches the renders gives sharper,
better-aligned results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import functools
import hashlib

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from scene_recon.data.base import SceneDataset
from scene_recon.registry import DATASETS, ConfigError
from scene_recon.structures import Cameras, SceneViews


@DATASETS.register("srn")
class SRNDataset(SceneDataset):
    """One split of ShapeNet-SRN, read straight from the extracted files."""

    def __init__(self, split_dir: str, near: float, far: float, pixel_centers: str) -> None:
        """
        Args:
            split_dir: Directory containing one sub-directory per scene, for
                example ``/data/srn_cars/cars_train``.
            near: Distance from the camera at which objects start. The
                dataset files do not store it; 0.8 is the usual value for cars.
            far: Distance from the camera at which objects end; 1.8 for cars.
            pixel_centers: How to read the principal point in the intrinsics
                files: ``"half_integer"`` or ``"integer"``; see the module docstring.
        """
        if pixel_centers not in _PRINCIPAL_POINT_SHIFT:
            raise ConfigError(
                f"pixel_centers must be one of {sorted(_PRINCIPAL_POINT_SHIFT)}, got {pixel_centers!r}."
            )
        self.split_dir = Path(split_dir)
        self.near = near
        self.far = far
        self.pixel_centers = pixel_centers

        if not self.split_dir.is_dir():
            raise ConfigError(f"SRN split directory does not exist: {self.split_dir}")

        self.scene_dirs = sorted(
            path for path in self.split_dir.iterdir() if (path / "intrinsics.txt").is_file()
        )
        if not self.scene_dirs:
            raise ConfigError(f"No scenes (folders with an intrinsics.txt) found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def num_views(self, scene_index: int) -> int:
        return len(self._image_paths(scene_index))

    def load_scene(self, scene_index: int, view_indices: Sequence[int] | None = None) -> SceneViews:
        scene_dir = self.scene_dirs[scene_index]
        image_paths = self._image_paths(scene_index)
        if view_indices is None:
            view_indices = range(len(image_paths))

        images = torch.stack([_read_image(image_paths[index]) for index in view_indices])
        poses = torch.stack(
            [_read_pose(scene_dir / "pose" / f"{image_paths[index].stem}.txt") for index in view_indices]
        )
        image_height, image_width = images.shape[-2:]
        intrinsics = _read_intrinsics(
            scene_dir / "intrinsics.txt", image_height, image_width, _PRINCIPAL_POINT_SHIFT[self.pixel_centers]
        )

        return SceneViews(
            images=images[None],
            cameras=Cameras(
                camera_to_world=poses[None],
                intrinsics=intrinsics.expand(1, len(poses), 3, 3).clone(),
            ),
            near=torch.tensor([self.near]),
            far=torch.tensor([self.far]),
            scene_ids=[scene_dir.name],
        )

    def fingerprint(self) -> dict:
        """Identifies the files of this split: every file's name and size, and every intrinsics file's content.

        Computed once per split directory and process; it reads only file
        metadata and the small intrinsics files, not the images.
        """
        return {
            "dataset": "srn",
            "num_scenes": len(self),
            "files_sha256": _split_digest(str(self.split_dir.resolve())),
        }

    def _image_paths(self, scene_index: int) -> list[Path]:
        return sorted((self.scene_dirs[scene_index] / "rgb").glob("*.png"))


def _read_image(path: Path) -> Tensor:
    """Read an image as a float tensor of shape (3, H, W) in [0, 1]."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1)


def _read_pose(path: Path) -> Tensor:
    """Read a camera-to-world matrix of shape (4, 4)."""
    return torch.from_numpy(np.loadtxt(path, dtype=np.float32).reshape(4, 4))


# Added to the stored principal point to express it in the pipeline's convention.
_PRINCIPAL_POINT_SHIFT = {"half_integer": 0.0, "integer": 0.5}


@functools.lru_cache(maxsize=8)
def _split_digest(split_dir: str) -> str:
    digest = hashlib.sha256()
    root = Path(split_dir)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(f"{path.relative_to(root)}\t{path.stat().st_size}\n".encode())
            if path.name == "intrinsics.txt":
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_intrinsics(path: Path, image_height: int, image_width: int, principal_point_shift: float) -> Tensor:
    """Read the intrinsics matrix of shape (3, 3) shared by all views of a scene.

    The file records the resolution the intrinsics were calibrated for. If the
    images on disk have a different resolution (some releases ship resized
    images), the intrinsics are scaled to match. The principal point is first
    converted to the pipeline's pixel convention, in which scaling is exact.
    """
    with open(path) as file:
        lines = [line for line in file.read().splitlines() if line.strip()]
    focal, center_x, center_y, _ = (float(value) for value in lines[0].split())
    calibrated_height, calibrated_width = (int(float(value)) for value in lines[-1].split())
    center_x += principal_point_shift
    center_y += principal_point_shift

    scale_x = image_width / calibrated_width
    scale_y = image_height / calibrated_height
    return torch.tensor(
        [
            [focal * scale_x, 0.0, center_x * scale_x],
            [0.0, focal * scale_y, center_y * scale_y],
            [0.0, 0.0, 1.0],
        ]
    )
