"""Tests of the Hiera-T encoder and the HieraNeRF model."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scene_recon.build import build_model
from scene_recon.config import Config, load_config
from scene_recon.geometry import sample_features_at_pixels
from scene_recon.models.base import pretrained_weights_of, weights_sha256
from scene_recon.models.contract import check_model_contract
from scene_recon.models.hiera.attention import HieraRayTransformer
from scene_recon.models.hiera.bridge import HieraSceneEncoding
from scene_recon.models.hiera.encoder import (
    DropPath,
    HieraBackbone,
    HieraEncoder,
    _max_pool_tokens,
    reroll,
    token_order,
    unroll,
)
from scene_recon.models.pixelnerf.encoders import ConvNeXtV2Encoder
from scene_recon.models.pixelnerf.model import PixelAlignedNeRF, _SceneEncoding
from scene_recon.models.testing.tiny_encoder import TinyConvEncoder
from scene_recon.registry import ENCODERS, MODELS, ConfigError
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
    encoder = HieraEncoder(weights=None, num_features=16)
    assert encoder.adapter.in_channels == 96 + 192 + 384
    assert encoder(torch.rand(2, 3, 64, 96)).shape == (2, 16, 16, 24)


def test_encoder_rejects_images_that_are_not_whole_mask_units():
    with pytest.raises(ValueError, match="multiples of 32"):
        HieraEncoder(weights=None, num_features=16)(torch.rand(1, 3, 48, 64))


def test_the_weights_must_be_named_in_the_config():
    # Leaving them out must not silently start from random weights; null does that on purpose.
    with pytest.raises(ConfigError, match="weights"):
        ENCODERS.build({"name": "hiera", "num_features": 16})


@pytest.mark.parametrize("rate", [-0.1, 1.0, float("nan")])
def test_stochastic_depth_rate_must_be_at_least_0_and_below_1(rate: float):
    with pytest.raises(ConfigError, match="drop_path_rate"):
        HieraEncoder(weights=None, drop_path_rate=rate)


def test_a_stochastic_depth_rate_just_below_1_stays_below_1_in_every_block():
    # In float32 the last block's rate of 0.99999999 would round to 1, and dropping with
    # probability 1 divides by zero.
    encoder = HieraEncoder(weights=None, num_features=16, drop_path_rate=0.99999999)
    rates = [block.drop_path.probability for block in encoder.backbone.blocks if isinstance(block.drop_path, DropPath)]

    assert rates and max(rates) < 1.0


def test_official_checkpoint_loads_and_unused_entries_are_ignored(tmp_path: Path):
    source = HieraBackbone()
    state = dict(source.state_dict())
    # Stage 4, not built. Hiera-T's block 10 starts it: twice the channels (768), and a projection.
    state["blocks.10.attn.qkv.weight"] = torch.zeros(3 * 768, 384)
    state["blocks.10.proj.weight"] = torch.zeros(768, 384)
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


def test_weights_of_a_hiera_with_a_longer_third_stage_are_rejected(tmp_path: Path):
    # Hiera-S has the same first ten blocks as Hiera-T, with the same shapes, but
    # its block 10 continues the third stage (no projection) where Hiera-T's
    # starts the fourth. Loading it would silently train a truncated Hiera-S.
    state = dict(HieraBackbone().state_dict())
    state.update({key.replace("blocks.9.", "blocks.10.", 1): value for key, value in state.items() if key.startswith("blocks.9.")})
    path = tmp_path / "hiera_small.pth"
    torch.save(state, path)

    with pytest.raises(ConfigError, match="third stage"):
        HieraEncoder(weights=str(path))


def test_weights_whose_next_block_widens_load_without_its_projection(tmp_path: Path):
    # A partial export: block 10's projection is missing, but its attention has the
    # width of the fourth stage (3 x 768 outputs from 384 inputs), so it starts that stage.
    state = dict(HieraBackbone().state_dict())
    state["blocks.10.attn.qkv.weight"] = torch.zeros(3 * 768, 384)
    path = tmp_path / "hiera.pth"
    torch.save(state, path)

    HieraEncoder(weights=str(path), num_features=16)


def test_weights_whose_next_block_shows_its_width_by_the_attention_bias_load(tmp_path: Path):
    state = dict(HieraBackbone().state_dict())
    state["blocks.10.attn.qkv.bias"] = torch.zeros(3 * 768)  # the fourth stage's width; the weight is missing
    path = tmp_path / "hiera.pth"
    torch.save(state, path)

    HieraEncoder(weights=str(path), num_features=16)


@pytest.mark.parametrize(
    "later_entries",
    [
        {"blocks.10.norm1.weight": torch.ones(384)},  # the same in both stages
        {"blocks.10.proj.weight": torch.zeros(768, 384)},  # a projection, but no attention to confirm the width
        {"blocks.10.proj.weight": None},
        {"blocks.10.attn.qkv.weight": torch.zeros(3 * 768, 768)},  # wider, but not from the third stage's width
        {"blocks.11.attn.qkv.weight": torch.zeros(3 * 384, 384)},  # block 10 missing, block 11 still in stage 3
        # A weight entry that is there but not a tensor is not replaced by a good bias.
        {"blocks.10.attn.qkv.weight": None, "blocks.10.attn.qkv.bias": torch.zeros(3 * 768)},
        {"blocks.10.attn.qkv.weight": "qkv", "blocks.10.attn.qkv.bias": torch.zeros(3 * 768)},
    ],
    ids=[
        "norm_only",
        "projection_only",
        "projection_none",
        "malformed_attention",
        "gap_before_stage_3_block",
        "weight_none_bias_fine",
        "weight_text_bias_fine",
    ],
)
def test_weights_that_do_not_show_where_the_third_stage_ends_are_refused(later_entries: dict, tmp_path: Path):
    state = {**HieraBackbone().state_dict(), **later_entries}
    path = tmp_path / "hiera.pth"
    torch.save(state, path)

    with pytest.raises(ConfigError, match="does not show whether"):
        HieraEncoder(weights=str(path))


def test_the_refusal_asks_for_weights_matching_the_configured_hiera(tmp_path: Path):
    # Configured as Hiera-S (11 blocks in stage 3), the advice must not point at Hiera-T weights.
    state = {**HieraBackbone(stage_depths=(1, 2, 11)).state_dict(), "blocks.14.norm1.weight": torch.ones(384)}
    path = tmp_path / "hiera_small.pth"
    torch.save(state, path)

    with pytest.raises(ConfigError, match="matches the encoder's settings") as error:
        HieraEncoder(weights=str(path), stage_depths=[1, 2, 11])
    assert "Hiera-T model" not in str(error.value)


def test_weights_of_another_width_report_the_shape_mismatch_first(tmp_path: Path):
    # Complete Hiera-T weights loaded into an encoder with the wrong embed_dim: the
    # error must point at the setting, not claim a longer third stage.
    state = dict(HieraBackbone().state_dict())
    state["blocks.10.attn.qkv.weight"] = torch.zeros(3 * 768, 384)
    path = tmp_path / "hiera.pth"
    torch.save(state, path)

    with pytest.raises(ConfigError, match="do not fit the encoder's shape.*embed_dim"):
        HieraEncoder(weights=str(path), embed_dim=64)


def test_a_weights_file_that_is_not_a_state_dict_is_a_config_error(tmp_path: Path):
    path = tmp_path / "list.pth"
    torch.save([1, 2, 3], path)

    with pytest.raises(ConfigError, match="not a state dict"):
        HieraEncoder(weights=str(path))


def test_a_weights_entry_that_is_not_a_tensor_is_a_config_error(tmp_path: Path):
    state = dict(HieraBackbone().state_dict())
    state["pos_embed"] = 3
    path = tmp_path / "hiera.pth"
    torch.save(state, path)

    with pytest.raises(ConfigError, match="not a tensor: pos_embed"):
        HieraEncoder(weights=str(path))


# ------------------------------------------------------------------ feature alignment


def _coordinate_ramp(size: int) -> Tensor:
    """An image whose red value is each pixel's x coordinate and green its y (pixel centres at i + 0.5)."""
    centres = torch.arange(size, dtype=torch.float32) + 0.5
    return torch.stack([centres.expand(size, size), centres[:, None].expand(size, size), torch.zeros(size, size)])


def _token_centres(encoder: HieraEncoder, size: int) -> Tensor:
    """A feature map holding the image coordinates (x, y) each Hiera token is computed around. Shape (1, 2, size/4, size/4)."""
    convolution = encoder.backbone.patch_embed.proj
    with torch.no_grad():
        convolution.weight.zero_()
        convolution.bias.zero_()
        centre = convolution.kernel_size[0] // 2
        convolution.weight[0, 0, centre, centre] = 1.0  # copy the red value of the kernel's centre pixel
        convolution.weight[1, 1, centre, centre] = 1.0  # and its green value
        return encoder.backbone.patch_embed(_coordinate_ramp(size)[None])[:, :2]


def _camera_seeing(pixels: Tensor, size: int) -> tuple[Cameras, Tensor]:
    """A camera at the origin looking along +z, and points at depth 1 that it sees at ``pixels`` (N, 2)."""
    focal, centre = 100.0, size / 2
    intrinsics = torch.tensor([[focal, 0, centre], [0, focal, centre], [0, 0, 1]])
    cameras = Cameras(torch.eye(4).expand(1, 1, 4, 4).clone(), intrinsics.expand(1, 1, 3, 3).clone())
    points = torch.cat([(pixels - centre) / focal, torch.ones(len(pixels), 1)], dim=-1)
    return cameras, points


# Points in the image interior, away from token centres and from the border.
_PIXELS = torch.tensor([[40.5, 64.5], [42.0, 10.0], [100.25, 77.0]])


def test_feature_offset_lines_hiera_tokens_up_with_the_pixels_they_were_computed_from():
    encoder = HieraEncoder(weights=None, num_features=16)
    tokens = _token_centres(encoder, 128)

    sampled = sample_features_at_pixels(tokens[:, None], (_PIXELS + encoder.feature_offset)[None, None], (128, 128))

    assert torch.allclose(sampled[0, 0], _PIXELS, atol=1e-4)


def test_hiera_nerf_bridge_reads_features_aligned_and_colours_at_the_pixel_itself(hiera_config: Config):
    # The bridge of the real model, so the test also covers the model handing it the encoder's offset.
    model = build_model(hiera_config)
    with torch.no_grad():
        model.bridge.feature_projection.weight.zero_()
        model.bridge.feature_projection.weight[:2, :2] = torch.eye(2)  # pass the x and y channels through
        model.bridge.feature_projection.bias.zero_()
    tokens = _token_centres(model.encoder, 128)  # (1, 2, 32, 32)
    unused_channels = tokens.new_zeros(1, model.encoder.num_features - 2, 32, 32)
    # The last point is just inside the right edge: the photo sees it, though shifted
    # by the feature offset it would lie outside.
    pixels = torch.cat([_PIXELS, torch.tensor([[127.8, 64.5]])])
    cameras, points = _camera_seeing(pixels, 128)
    scene = HieraSceneEncoding(
        feature_maps=torch.cat([tokens, unused_channels], dim=1)[:, None],
        images=_coordinate_ramp(128)[None, None],
        cameras=cameras,
        image_size=(128, 128),
        near=torch.tensor([0.5]),
        far=torch.tensor([1.5]),
    )

    with torch.no_grad():
        output = model.bridge(scene, points[None, :, None], torch.tensor([[[0.0, 0.0, 1.0]]]).expand(1, len(pixels), 3))

    interior = len(_PIXELS)
    assert torch.allclose(output.view_features[0, :interior, 0, :2], _PIXELS, atol=1e-4)  # features: shifted onto the tokens
    assert torch.allclose(output.view_colors[0, :interior, 0, :2], _PIXELS, atol=1e-4)  # colours: the photo's own pixels
    assert output.visible.all()  # visibility: the photo's own pixels too


class _RecordFeatures(nn.Module):
    """Stands in for a decoder and keeps the features it is given."""

    def forward(self, features: Tensor, points_camera: Tensor, directions_camera: Tensor):
        self.features = features
        batch_size, _, num_points, _ = features.shape
        return features.new_zeros(batch_size, num_points, 3), features.new_zeros(batch_size, num_points)


def test_pixel_aligned_nerf_reads_hiera_features_aligned():
    model = PixelAlignedNeRF(
        encoder={"name": "hiera", "weights": None, "num_features": 2},
        decoder={"name": "pixelnerf_mlp", "hidden_width": 8},
    )
    model.decoder = _RecordFeatures()
    cameras, points = _camera_seeing(_PIXELS, 128)
    scene = _SceneEncoding(_token_centres(model.encoder, 128)[:, None], cameras, (128, 128))

    with torch.no_grad():
        model.query_field(scene, points[None, :, None], torch.tensor([[[0.0, 0.0, 1.0]]]).expand(1, len(_PIXELS), 3), "coarse")

    assert torch.allclose(model.decoder.features[0, 0], _PIXELS, atol=1e-4)


# A constant feature_offset holds only when every stride divides the image evenly.
@pytest.mark.parametrize(
    ("build_encoder", "size", "multiple"),
    [
        (lambda: ConvNeXtV2Encoder(model_name="convnextv2_atto", pretrained=False, num_features=8), 80, 32),
        (lambda: TinyConvEncoder(num_features=4), 65, 2),
    ],
    ids=["convnextv2", "tiny_conv"],
)
def test_encoders_reject_image_sides_their_strides_do_not_divide(build_encoder, size: int, multiple: int):
    encoder = build_encoder()

    with pytest.raises(ValueError, match=f"multiples of {multiple}"):
        encoder(torch.rand(1, 3, size, size))
    encoder(torch.rand(1, 3, 2 * multiple, 2 * multiple))  # a size they divide is fine


_MODELS_WITH_AN_ENCODER = [
    {"name": "pixel_aligned_nerf", "decoder": {"name": "pixelnerf_mlp", "hidden_width": 8}},
    {"name": "tiny_pixel_aligned_nerf", "decoder": {"name": "tiny_pooled_mlp", "hidden_width": 8}},
    {"name": "hiera_nerf", "width": 8, "decoder": {"name": "hiera_ray_attention", "num_heads": 2}},
]


_ENCODER_ATTRIBUTES = {"num_features": 4, "feature_offset": 0.0, "adapter": None, "pretrained_weights": None}


def _encoder_class(attributes: dict) -> type:
    """An encoder plug-in with the given class attributes."""

    def forward(self, images: Tensor) -> Tensor:
        return images.new_zeros(len(images), 4, images.shape[-2] // 4, images.shape[-1] // 4)

    return type("TestEncoder", (nn.Module,), {**attributes, "forward": forward})


def _encoder_without(attribute: str) -> type:
    """An encoder plug-in with every attribute that build_encoder checks, except one."""
    return _encoder_class({name: value for name, value in _ENCODER_ATTRIBUTES.items() if name != attribute})


@pytest.mark.parametrize("attribute", ["num_features", "feature_offset", "adapter", "pretrained_weights"])
@pytest.mark.parametrize("model", _MODELS_WITH_AN_ENCODER, ids=lambda model: model["name"])
def test_every_model_names_an_encoder_missing_an_attribute(model: dict, attribute: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(ENCODERS._classes, "incomplete", _encoder_without(attribute))

    with pytest.raises(ConfigError, match=f"'incomplete' has no attribute '{attribute}'"):
        MODELS.build({**model, "encoder": {"name": "incomplete"}})


_A_HASH = "0" * 64


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("adapter", "conv"),
        ("pretrained_weights", "weights/encoder.pth"),
        ("pretrained_weights", {"source": Path("weights/encoder.pth"), "sha256": _A_HASH}),
        ("pretrained_weights", {"source": "weights/encoder.pth", "sha256": "not a hash"}),
        ("pretrained_weights", {"source": "weights/encoder.pth", "sha256": _A_HASH, "step": 0}),
    ],
    ids=["adapter_not_a_module", "weights_not_a_dict", "source_not_a_string", "hash_malformed", "extra_entry"],
)
def test_an_encoder_whose_adapter_or_weights_record_is_malformed_is_refused(
    attribute: str, value, monkeypatch: pytest.MonkeyPatch
):
    """Review F44: caught when the model is built, not when the checkpoint is read back."""
    monkeypatch.setitem(ENCODERS._classes, "malformed", _encoder_class({**_ENCODER_ATTRIBUTES, attribute: value}))

    with pytest.raises(ConfigError, match="The encoder 'malformed' has"):
        MODELS.build({**_MODELS_WITH_AN_ENCODER[0], "encoder": {"name": "malformed"}})


def test_a_malformed_weights_record_outside_the_encoder_is_refused_too():
    part = nn.Linear(2, 2)
    part.pretrained_weights = {"source": "weights/part.pth"}

    with pytest.raises(ConfigError, match="part '1' has pretrained_weights"):
        pretrained_weights_of(nn.Sequential(nn.Identity(), part))


def test_the_weights_hash_works_for_every_data_type_and_ignores_creation_order():
    """Review F48."""
    first, second = nn.Linear(2, 2), nn.Linear(2, 2)
    one_way, other_way = nn.Module(), nn.Module()
    one_way.a, one_way.b = first, second
    other_way.b, other_way.a = second, first

    in_float32 = weights_sha256(one_way)
    assert weights_sha256(other_way) == in_float32

    one_way.to(torch.bfloat16)  # other_way holds the same layers
    assert weights_sha256(one_way) == weights_sha256(other_way) != in_float32


def test_the_hiera_encoder_records_the_weights_it_loaded(tmp_path: Path):
    """Review B(b): the hash identifies the loaded backbone, whatever the random start of the adapter."""
    state = dict(HieraBackbone().state_dict())
    path = tmp_path / "hiera.pth"
    torch.save(state, path)
    first, second = HieraEncoder(weights=str(path), num_features=16), HieraEncoder(weights=str(path), num_features=16)

    assert first.pretrained_weights == {"source": str(path), "sha256": weights_sha256(first.backbone)}
    assert second.pretrained_weights == first.pretrained_weights
    assert not torch.equal(first.adapter.weight, second.adapter.weight)

    name = next(iter(state))
    torch.save({**state, name: state[name] + 1.0}, path)  # another version of the file
    assert HieraEncoder(weights=str(path), num_features=16).pretrained_weights["sha256"] != first.pretrained_weights["sha256"]
    assert HieraEncoder(weights=None, num_features=16).pretrained_weights is None
    assert ConvNeXtV2Encoder(model_name="convnextv2_atto", pretrained=False, num_features=8).pretrained_weights is None


def test_the_depth_feature_measures_distance_along_the_ray_like_near_and_far():
    """Review F4 and F51: on a slanted ray, the points at the near and far bounds get exactly 0 and 1."""
    from scene_recon.models.hiera.bridge import HieraNeRFBridge

    bridge = HieraNeRFBridge(num_features=4, width=8, position_frequencies=2, feature_offset=1.5)
    cameras, _ = _camera_seeing(torch.zeros(0, 2), 32)  # the photo's camera sits at the origin
    scene = HieraSceneEncoding(
        feature_maps=torch.zeros(1, 1, 4, 8, 8),
        images=torch.zeros(1, 1, 3, 32, 32),
        cameras=cameras,
        image_size=(32, 32),
        near=torch.tensor([1.0]),
        far=torch.tensor([2.0]),
    )
    direction = torch.tensor([0.6, 0.0, 0.8])  # a unit-length ray from that camera, away from the image centre
    points = torch.stack([1.0 * direction, 1.5 * direction, 2.0 * direction])

    with torch.no_grad():
        output = bridge(scene, points[None, None], direction.expand(1, 1, 3))

    depth = output.view_geometry[0, :, 0, -1]  # the last geometry entry, for the one photo
    assert torch.allclose(depth, torch.tensor([0.0, 0.5, 1.0]))


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


_SMALL_HIERA = {"name": "hiera", "weights": None, "num_features": 8, "embed_dim": 8, "num_heads": 1}


@pytest.mark.parametrize(
    "model",
    [
        {
            "name": "pixel_aligned_nerf",
            "encoder": {"name": "convnextv2", "model_name": "convnextv2_atto", "pretrained": False, "num_features": 8},
            "decoder": {"name": "pixelnerf_mlp", "hidden_width": 8},
        },
        {"name": "pixel_aligned_nerf", "encoder": _SMALL_HIERA, "decoder": {"name": "pixelnerf_mlp", "hidden_width": 8}},
        {"name": "hiera_nerf", "width": 8, "encoder": _SMALL_HIERA, "decoder": {"name": "hiera_ray_attention", "num_heads": 2}},
    ],
    ids=["convnextv2_mlp", "hiera_mlp", "hiera_nerf"],
)
def test_the_new_fusion_layer_learns_at_the_base_rate_in_every_model(model: dict):
    """Review B(a): a lower rate for the pretrained encoder must not slow down its randomly started 1x1 layer."""
    from scene_recon.config import OptimizationSettings
    from scene_recon.optim import build_optimizer_and_scheduler

    built = MODELS.build(model)
    settings = OptimizationSettings(
        optimizer="adam", betas=[0.9, 0.999], weight_decay=0.0, learning_rate=1.0e-4, warmup_steps=0,
        decay_rate=0.1, decay_steps=1000, group_overrides={"encoder": {"learning_rate": 1.0e-5, "warmup_steps": 2000}},
    )
    optimizer, _ = build_optimizer_and_scheduler(built, settings)
    groups = {group["name"]: group for group in optimizer.param_groups}

    assert {id(p) for p in groups["adapter"]["params"]} == {id(p) for p in built.encoder.adapter.parameters()}
    assert groups["adapter"]["initial_lr"] == 1.0e-4 and groups["adapter"]["lr"] == 1.0e-4  # no warmup
    assert groups["encoder"]["initial_lr"] == 1.0e-5
    adapter_ids = {id(p) for p in built.encoder.adapter.parameters()}
    assert not adapter_ids & {id(p) for p in groups["encoder"]["params"]}


def _built_at_seed(model: dict, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return MODELS.build(model)


def _same_weights(first: nn.Module, second: nn.Module) -> bool:
    first_state, second_state = first.state_dict(), second.state_dict()
    return first_state.keys() == second_state.keys() and all(
        torch.equal(first_state[name], second_state[name]) for name in first_state
    )


_MLP = {"name": "pixelnerf_mlp", "hidden_width": 8}
_CONVNEXT = {"name": "convnextv2", "model_name": "convnextv2_atto", "pretrained": False, "num_features": 8}


def test_the_mlp_decoder_starts_the_same_whichever_encoder_is_built_before_it():
    """Review F8: in the encoder ablation the runs differ only in their encoder, also before the first step."""
    after_convnext = _built_at_seed({"name": "pixel_aligned_nerf", "encoder": _CONVNEXT, "decoder": _MLP}, seed=0)
    after_hiera = _built_at_seed({"name": "pixel_aligned_nerf", "encoder": _SMALL_HIERA, "decoder": _MLP}, seed=0)
    other_seed = _built_at_seed({"name": "pixel_aligned_nerf", "encoder": _CONVNEXT, "decoder": _MLP}, seed=1)

    assert _same_weights(after_convnext.decoder, after_hiera.decoder)
    assert not _same_weights(after_convnext.decoder, other_seed.decoder)  # it still follows the seed


def test_a_shared_hiera_encoder_starts_the_same_in_both_hiera_models():
    """Review F43: hiera_mlp and hiera_nerf differ in what follows the encoder, not in the encoder's start."""
    hiera_mlp = _built_at_seed({"name": "pixel_aligned_nerf", "encoder": _SMALL_HIERA, "decoder": _MLP}, seed=0)
    hiera_nerf = _built_at_seed(
        {"name": "hiera_nerf", "width": 8, "encoder": _SMALL_HIERA, "decoder": {"name": "hiera_ray_attention", "num_heads": 2}},
        seed=0,
    )

    assert _same_weights(hiera_mlp.encoder, hiera_nerf.encoder)  # backbone and adapter


def test_a_part_with_its_own_random_numbers_leaves_the_global_generator_alone():
    from scene_recon.models.base import part_seed, seeded_random_numbers

    torch.manual_seed(0)
    before = torch.get_rng_state()
    with seeded_random_numbers(part_seed("decoder")):
        inside = torch.rand(3)
    assert torch.equal(torch.get_rng_state(), before)
    assert not torch.equal(inside, torch.rand(3))  # the part's numbers are not the global ones


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
