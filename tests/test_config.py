import re
from pathlib import Path

import pytest
import yaml

from scene_recon.config import load_yaml_with_bases, load_config
from scene_recon.build import build_metrics
from scene_recon.metrics import PSNR
from scene_recon.registry import DATASETS, DECODERS, ENCODERS, LOSSES, METRICS, MODELS, ConfigError

from conftest import CONFIG_DIR

RUNNABLE_CONFIGS = ["smoke_test.yaml", "srn_cars_tiny_model.yaml"]


@pytest.mark.parametrize("file_name", RUNNABLE_CONFIGS)
def test_shipped_configs_are_valid(file_name: str):
    load_config(CONFIG_DIR / file_name)


# The registry each component section's ``name`` is looked up in.
_REGISTRY_OF_SECTION = {
    "model": MODELS,
    "encoder": ENCODERS,
    "decoder": DECODERS,
    "loss": LOSSES,
    "train": DATASETS,
    "val": DATASETS,
    "test": DATASETS,
}


def _named_components(node, section=None):
    """Every (section, name) of a component section anywhere in a parsed YAML file."""
    if isinstance(node, dict):
        if section in _REGISTRY_OF_SECTION and "name" in node:
            yield section, node["name"]
        for key, child in node.items():
            yield from _named_components(child, key)
    elif isinstance(node, list):
        for child in node:
            yield from _named_components(child, section)


@pytest.mark.parametrize(
    "path", sorted(CONFIG_DIR.rglob("*.yaml")), ids=lambda path: str(path.relative_to(CONFIG_DIR))
)
def test_every_component_a_shipped_config_names_is_registered(path: Path):
    """A renamed or misspelt component fails here, not in the preflight of a long suite."""
    named = list(_named_components(yaml.safe_load(path.read_text()) or {}))
    unknown = [f"{section}: {name}" for section, name in named if name not in _REGISTRY_OF_SECTION[section].names()]

    assert not unknown, f"{path.relative_to(CONFIG_DIR)} names components that are not registered: {unknown}"


def test_child_config_overrides_its_bases():
    config = load_config(CONFIG_DIR / "smoke_test.yaml")

    assert config.experiment.run_name == "smoke_test"  # set in smoke_test.yaml
    assert config.optimization.betas == [0.9, 0.999]  # inherited from base.yaml
    assert config.model["name"] == "tiny_pixel_aligned_nerf"  # inherited from the model file


def test_command_line_overrides_are_parsed_as_yaml():
    config = load_config(
        CONFIG_DIR / "smoke_test.yaml",
        overrides=["training.batch_size=7", "evaluation.metrics=[psnr]", "model.decoder.hidden_width=8"],
    )

    assert config.training.batch_size == 7
    assert config.evaluation.metrics == ["psnr"]
    assert config.model["decoder"]["hidden_width"] == 8


def test_misspelt_setting_is_rejected():
    with pytest.raises(ConfigError, match="batch_sise"):
        load_config(CONFIG_DIR / "smoke_test.yaml", overrides=["training.batch_sise=7"])


def test_missing_setting_is_rejected(tmp_path: Path):
    content = load_config(CONFIG_DIR / "smoke_test.yaml").raw
    del content["training"]["num_steps"]
    path = tmp_path / "incomplete.yaml"
    path.write_text(yaml.safe_dump(content))

    with pytest.raises(ConfigError, match="num_steps"):
        load_config(path)


def test_number_written_without_decimal_point_is_still_a_number():
    # YAML parses "1e-4" as a string; the loader must turn it into a float.
    config = load_config(CONFIG_DIR / "smoke_test.yaml", overrides=["optimization.learning_rate=1e-4"])

    assert config.optimization.learning_rate == pytest.approx(1e-4)


def test_selecting_another_component_drops_the_previous_arguments(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"model": {"name": "first", "only_for_first": 1}}))
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"base": "base.yaml", "model": {"name": "second", "width": 2}}))

    assert load_yaml_with_bases(child)["model"] == {"name": "second", "width": 2}


def test_evaluation_protocols_must_differ_in_number_of_views():
    with pytest.raises(ConfigError, match="same number of views"):
        load_config(CONFIG_DIR / "smoke_test.yaml", overrides=["evaluation.source_view_sets=[[0], [5]]"])


def test_missing_base_file_is_named_in_the_error(tmp_path: Path):
    path = tmp_path / "child.yaml"
    path.write_text(yaml.safe_dump({"base": "nowhere.yaml"}))

    with pytest.raises(ConfigError, match="nowhere.yaml"):
        load_config(path)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("training.num_steps=-1", "training.num_steps must be at least 1"),
        ("training.num_steps=2.9", "must be a whole number"),
        ("training.num_steps=true", "must be a whole number"),
        ("evaluation.save_images=1", "must be true or false"),
        ("evaluation.source_view_sets=[]", "must contain at least one set of views"),
        ("evaluation.source_view_sets=[[0, 0]]", "lists a view more than once"),
        ("evaluation.num_scenes=0", "evaluation.num_scenes must be at least 1"),
        ("validation.source_views=[]", "must contain at least one view index"),
        ("renderer.num_coarse_samples=1", "renderer.num_coarse_samples must be at least 2"),
        ("renderer.num_coarse_samples=2", "at least 3 when the fine pass is enabled"),
        ("optimization.learning_rate=0", "learning_rate must be positive"),
        ("optimization.betas=[0.9]", "betas must be two values"),
    ],
)
def test_invalid_values_are_rejected(override: str, message: str):
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(CONFIG_DIR / "smoke_test.yaml", overrides=[override])


def test_every_invalid_value_is_reported_at_once():
    with pytest.raises(ConfigError) as error:
        load_config(CONFIG_DIR / "smoke_test.yaml", overrides=["training.num_steps=0", "training.batch_size=0"])

    assert "training.num_steps" in str(error.value) and "training.batch_size" in str(error.value)


def test_whole_number_written_in_scientific_notation_is_accepted():
    config = load_config(CONFIG_DIR / "smoke_test.yaml", overrides=["training.num_steps=4.0e+2"])

    assert config.training.num_steps == 400 and isinstance(config.training.num_steps, int)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("{encoder: {learning_rate: .nan}}", "must be a finite number"),
        ("{encoder: {learning_rate: -0.001}}", "learning_rate must be positive"),
        ("{encoder: {learning_rate: true}}", "must be a finite number"),
        ("{encoder: {warmup_steps: 2.9}}", "must be a whole number"),
        ("{encoder: {momentum: 0.9}}", "unknown key"),
    ],
)
def test_invalid_group_overrides_are_rejected(override: str, message: str):
    with pytest.raises(ConfigError, match=message):
        load_config(CONFIG_DIR / "smoke_test.yaml", overrides=[f"optimization.group_overrides={override}"])


def test_valid_group_overrides_are_converted():
    config = load_config(
        CONFIG_DIR / "smoke_test.yaml",
        overrides=["optimization.group_overrides={encoder: {learning_rate: 1e-5, warmup_steps: 4.0e+2}}"],
    )

    assert config.optimization.group_overrides == {"encoder": {"learning_rate": 1e-5, "warmup_steps": 400}}


# A tiny metric with an option, registered once for the tests below; it needs no downloads.
if "scaled_psnr" not in METRICS.names():

    @METRICS.register("scaled_psnr")
    class ScaledPSNR(PSNR):
        def __init__(self, scale: float = 1.0) -> None:
            super().__init__()
            self.scale = scale


def test_metrics_can_be_names_or_mappings_with_options():
    config = load_config(
        CONFIG_DIR / "smoke_test.yaml",
        overrides=[
            "evaluation.metrics=[psnr, {name: scaled_psnr, scale: 2.0}]",
            "validation.metrics=[{name: lpips, network: alex}]",
        ],
    )

    metrics = build_metrics(config.evaluation.metrics)
    assert list(metrics) == ["psnr", "scaled_psnr"] and metrics["scaled_psnr"].scale == 2.0
    assert config.validation.metrics == [{"name": "lpips", "network": "alex"}]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("evaluation.metrics=[{network: alex}]", "must have a 'name'"),
        ("evaluation.metrics=[{name: 5}]", "must have a 'name'"),
        ("evaluation.metrics=[7]", "must be text or a mapping"),
        ("evaluation.metrics=[psnr, {name: psnr}]", "more than once"),
    ],
)
def test_malformed_metric_entries_are_rejected(override: str, message: str):
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(CONFIG_DIR / "smoke_test.yaml", overrides=[override])


def test_unknown_metric_option_is_reported_when_the_metric_is_built():
    config = load_config(CONFIG_DIR / "smoke_test.yaml", overrides=["evaluation.metrics=[{name: scaled_psnr, scal: 2}]"])

    with pytest.raises(ConfigError, match="scaled_psnr"):
        build_metrics(config.evaluation.metrics)
