"""Fixtures shared by the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scene_recon import environment
from scene_recon.config import Config, load_config

CONFIG_DIR = Path(__file__).parent.parent / "configs"

READ_CPU_MODEL = environment.cpu_model
"""The real reader of the CPU's model, for the test of it; every other test sees a fixed name."""


@pytest.fixture(autouse=True)
def named_cpu(monkeypatch):
    """A fixed CPU model, so evaluations on the CPU compare as the same even where the model cannot be read."""
    monkeypatch.setattr(environment, "cpu_model", lambda: "Test CPU")


@pytest.fixture
def smoke_config(tmp_path: Path) -> Config:
    """The smoke-test config, shrunk further and redirected to a temporary output directory."""
    return load_config(
        CONFIG_DIR / "smoke_test.yaml",
        overrides=[
            f"experiment.output_dir={tmp_path}",
            "experiment.device=cpu",
            "training.num_steps=6",
            "training.batch_size=2",
            "training.rays_per_scene=32",
            "training.log_every=2",
            "training.validate_every=3",
            "training.checkpoint_every=3",
            "validation.num_scenes=1",
            "validation.num_target_views=1",
            "evaluation.num_scenes=2",
            "evaluation.num_target_views=2",
            "data.train.image_size=16",
            "data.val.image_size=16",
            "data.test.image_size=16",
        ],
    )


# ---------------------------------------------------------------- an on-disk SRN dataset


def write_srn_scene(
    scene_dir: Path,
    num_views: int,
    image_size: int,
    calibrated_size: int | None = None,
    focal: float = 131.25,
) -> None:
    """Create a scene in the on-disk layout of ShapeNet-SRN, with recognisable content.

    Image ``view`` is filled with the grey level ``view * 10``, camera ``view``
    sits at x = ``view``. PNGs are stored uncompressed, so replacing an image
    with other pixels of the same size keeps the file size (as some tests need).
    """
    calibrated_size = calibrated_size or image_size
    (scene_dir / "rgb").mkdir(parents=True)
    (scene_dir / "pose").mkdir()
    center = calibrated_size / 2
    (scene_dir / "intrinsics.txt").write_text(
        f"{focal} {center} {center} 0.\n0. 0. 0.\n1.\n{calibrated_size} {calibrated_size}\n"
    )
    for view in range(num_views):
        write_srn_image(scene_dir / "rgb" / f"{view:06d}.png", image_size, grey_level=view * 10)
        pose = np.eye(4)
        pose[0, 3] = view
        (scene_dir / "pose" / f"{view:06d}.txt").write_text(" ".join(f"{value:.1f}" for value in pose.flatten()))


def write_srn_image(path: Path, image_size: int, grey_level: int) -> None:
    pixels = np.full((image_size, image_size, 3), grey_level, dtype=np.uint8)
    Image.fromarray(pixels).save(path, compress_level=0)


def write_srn_splits(root: Path, num_views: int = 6, image_size: int = 8) -> dict[str, Path]:
    """Train, validation and test splits of two scenes each. Returns the split directories."""
    splits = {}
    for split in ("train", "val", "test"):
        for scene in ("car_a", "car_b"):
            write_srn_scene(root / split / scene, num_views=num_views, image_size=image_size)
        splits[split] = root / split
    return splits


def srn_overrides(splits: dict[str, Path], output_dir: Path) -> list[str]:
    """Smoke-test overrides that train and evaluate on an on-disk SRN dataset from ``write_srn_splits``."""
    data = [
        f"data.{split}={{name: srn, split_dir: '{path}', near: 0.8, far: 1.8, pixel_centers: half_integer}}"
        for split, path in splits.items()
    ]
    return data + [
        f"experiment.output_dir={output_dir}",
        "experiment.device=cpu",
        "training.num_steps=2",
        "training.batch_size=2",
        "training.rays_per_scene=16",
        "training.views_per_scene=3",
        "training.num_source_views=[1, 2]",
        "training.log_every=1",
        "training.validate_every=2",
        "training.checkpoint_every=2",
        "validation.num_scenes=1",
        "validation.num_target_views=1",
        "validation.source_views=[0]",
        "evaluation.source_view_sets=[[0], [0, 2]]",
        "evaluation.num_target_views=1",
        "evaluation.metrics=[psnr]",
        "renderer.num_coarse_samples=6",
        "renderer.num_fine_samples=3",
        "profiling.num_frames=1",
        "profiling.num_warmup_frames=0",
    ]


@pytest.fixture
def srn_config(tmp_path: Path) -> Config:
    """The smoke-test model on a small on-disk SRN dataset under ``tmp_path / "srn"``."""
    splits = write_srn_splits(tmp_path / "srn")
    return load_config(CONFIG_DIR / "smoke_test.yaml", overrides=srn_overrides(splits, tmp_path / "runs"))
