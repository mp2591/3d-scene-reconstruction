from pathlib import Path

import pytest
import yaml

from scene_recon.config import _load_yaml_with_bases, load_config
from scene_recon.registry import ConfigError

from conftest import CONFIG_DIR

RUNNABLE_CONFIGS = ["smoke_test.yaml", "srn_cars_tiny_model.yaml"]


@pytest.mark.parametrize("file_name", RUNNABLE_CONFIGS)
def test_shipped_configs_are_valid(file_name: str):
    load_config(CONFIG_DIR / file_name)


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

    assert _load_yaml_with_bases(child)["model"] == {"name": "second", "width": 2}
