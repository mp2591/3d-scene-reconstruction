"""Tests of the Hiera-T encoder and the HieraNeRF model."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scene_recon.build import build_model
from scene_recon.config import Config, load_config
from scene_recon.models.contract import check_model_contract
from scene_recon.models.hiera.attention import HieraRayTransformer
from scene_recon.models.hiera.encoder import (
    HieraBackbone,
    HieraEncoder,
    _max_pool_tokens,
    reroll,
    token_order,
    unroll,
)
from scene_recon.registry import ConfigError
from scene_recon.structures import Cameras, SceneViews
from scene_recon.training import Trainer

CONFIG_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def hiera_config(tmp_path: Path) -> Config:
    """The HieraNeRF smoke test, shrunk to a few steps."""
    return load_config(
        CONFIG_DIR / "hiera_nerf_smoke_test.yaml",
        overrides=[
            f"experiment.output_dir={tmp_path}",
            "experiment.device=cpu",
            "training.num_steps=2",
            "training.batch_size=2",
            "training.rays_per_scene=16",
            "training.log_every=1",
            "training.validate_every=2",
            "training.checkpoint_every=2",
            "validation.num_scenes=1",
            "validation.num_target_views=1",
            "renderer.num_coarse_samples=6",
            "renderer.num_fine_samples=3",
        ],
    )


# ------------------------------------------------------------------ token order


def test_unrolled_tokens_reroll_to_the_original_grid():
    grid = (16, 24)
    image = torch.randn(2, 5, *grid)
    tokens = image.flatten(2).transpose(1, 2)

    assert torch.equal(reroll(unroll(tokens, grid), grid, num_pools=0), image)


def test_pooling_unrolled_tokens_is_max_pooling_the_image():
    grid = (16, 16)
    image = torch.randn(2, 5, *grid)
    tokens = unroll(image.flatten(2).transpose(1, 2), grid)

    once = _max_pool_tokens(tokens, 4)
    twice = _max_pool_tokens(once, 4)

    assert torch.equal(reroll(once, grid, num_pools=1), F.max_pool2d(image, 2))
    assert torch.equal(reroll(twice, grid, num_pools=2), F.max_pool2d(image, 4))


def test_mask_units_are_squares_of_8x8_tokens():
    grid = (32, 16)
    order = token_order(grid)
    num_units = grid[0] * grid[1] // 64
    for unit in range(num_units):
        members = order[unit::num_units]  # how MaskUnitAttention groups tokens
        rows, columns = members // grid[1], members % grid[1]
        assert len(members) == 64
        assert len(set((rows // 8).tolist())) == 1 and len(set((columns // 8).tolist())) == 1


# ------------------------------------------------------------------ encoder


def test_backbone_has_the_first_three_stages_of_hiera_t():
    backbone = HieraBackbone()
    assert backbone.stage_channels == [96, 192, 384]
    assert len(backbone.blocks) == 1 + 2 + 7
    assert [block.attn.heads for block in backbone.blocks] == [1, 2, 2, 4, 4, 4, 4, 4, 4, 4]


def test_backbone_stage_outputs_have_a_quarter_an_eighth_and_a_sixteenth_of_the_size():
    outputs = HieraBackbone()(torch.rand(2, 3, 64, 96))
    assert [tuple(output.shape) for output in outputs] == [(2, 96, 16, 24), (2, 192, 8, 12), (2, 384, 4, 6)]


def test_encoder_fuses_the_stages_into_one_map_at_a_quarter_of_the_size():
    encoder = HieraEncoder(num_features=16)
    assert encoder.adapter.in_channels == 96 + 192 + 384
    assert encoder(torch.rand(2, 3, 64, 96)).shape == (2, 16, 16, 24)


def test_encoder_rejects_images_that_are_not_whole_mask_units():
    with pytest.raises(ValueError, match="multiples of 32"):
        HieraEncoder(num_features=16)(torch.rand(1, 3, 48, 64))


def test_official_checkpoint_loads_and_unused_entries_are_ignored(tmp_path: Path):
    source = HieraBackbone()
    state = dict(source.state_dict())
    state["blocks.10.attn.qkv.weight"] = torch.zeros(1)  # stage 4, not built
    state["head.projection.weight"] = torch.zeros(1)  # classification head
    path = tmp_path / "hiera.pth"
    torch.save({"model_state": state, "epoch": 400}, path)

    encoder = HieraEncoder(weights=str(path), num_features=16)

    for name, value in source.state_dict().items():
        assert torch.equal(encoder.backbone.state_dict()[name], value), name


def test_checkpoint_missing_entries_is_a_config_error(tmp_path: Path):
    state = dict(HieraBackbone().state_dict())
    del state["blocks.9.mlp.fc2.weight"]
    path = tmp_path / "hiera.pth"
    torch.save(state, path)

    with pytest.raises(ConfigError, match="lack 1 entries"):
        HieraEncoder(weights=str(path))


def test_missing_checkpoint_file_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        HieraEncoder(weights=str(tmp_path / "nothing.pth"))


# ------------------------------------------------------------------ decoder


@pytest.mark.parametrize("num_samples", [1, 7, 8, 13, 64])
def test_ray_transformer_gives_one_finite_output_per_sample(num_samples: int):
    transformer = HieraRayTransformer(width=16, num_heads=2, window_size=8)
    output = transformer(torch.randn(3, num_samples, 16))
    assert output.shape == (3, num_samples, 16)
    assert torch.isfinite(output).all()


def test_ray_transformer_mixes_information_along_the_ray():
    transformer = HieraRayTransformer(width=16, num_heads=2, window_size=4).eval()
    tokens = torch.randn(1, 12, 16)
    changed = tokens.clone()
    changed[0, -1] += 1.0  # the last sample, in another local group than the first

    with torch.no_grad():
        difference = transformer(changed)[0, 0] - transformer(tokens)[0, 0]
    assert difference.abs().max() > 0  # reached through the global step


# ------------------------------------------------------------------ the model


def test_hiera_nerf_satisfies_the_model_contract(hiera_config: Config):
    check_model_contract(
        build_model(hiera_config),
        image_size=32,
        num_coarse_samples=6,
        num_fine_samples=3,
        inference_view_counts=(1, 2),
        training_view_counts=(1, 2, 3),
    )


def test_hiera_nerf_parameter_groups_separate_the_pretrained_encoder(hiera_config: Config):
    model = build_model(hiera_config)
    groups = {group.name: group.parameters for group in model.parameter_groups()}

    assert list(groups) == ["encoder", "adapter", "bridge", "decoder"]
    assert {id(p) for p in groups["encoder"]} == {id(p) for p in model.encoder.backbone.parameters()}


def test_points_no_photo_sees_still_get_a_finite_prediction(hiera_config: Config):
    model = build_model(hiera_config).eval()
    camera_to_world = torch.eye(4).expand(1, 2, 4, 4).clone()
    intrinsics = torch.tensor([[40.0, 0, 16], [0, 40.0, 16], [0, 0, 1]]).expand(1, 2, 3, 3)
    views = SceneViews(
        images=torch.rand(1, 2, 3, 32, 32),
        cameras=Cameras(camera_to_world, intrinsics),
        near=torch.tensor([0.8]),
        far=torch.tensor([1.8]),
        scene_ids=["behind"],
    )
    points = torch.tensor([0.0, 0.0, -1.0]).expand(1, 4, 5, 3)  # behind both cameras
    directions = torch.tensor([0.0, 0.0, -1.0]).expand(1, 4, 3)

    with torch.no_grad():
        field = model.query_field(model.encode_scene(views), points, directions, "coarse")

    assert torch.isfinite(field.rgb).all() and torch.isfinite(field.density).all()


def test_hiera_nerf_trains(hiera_config: Config):
    Trainer(hiera_config).train()
    assert (hiera_config.run_dir / "checkpoints" / "last.pt").is_file()
