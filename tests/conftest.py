"""Fixtures shared by the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from scene_recon.config import Config, load_config

CONFIG_DIR = Path(__file__).parent.parent / "configs"


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
