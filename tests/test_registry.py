import pytest

from scene_recon.registry import ConfigError, Registry


def _registry_with_one_widget() -> Registry:
    registry = Registry("widget")

    @registry.register("simple")
    class SimpleWidget:
        def __init__(self, size: int, color: str = "red") -> None:
            self.size = size
            self.color = color

    return registry


def test_build_passes_config_keys_to_the_constructor():
    widget = _registry_with_one_widget().build({"name": "simple", "size": 3})

    assert widget.size == 3
    assert widget.color == "red"


def test_build_accepts_arguments_supplied_by_the_pipeline():
    widget = _registry_with_one_widget().build({"name": "simple"}, size=5)

    assert widget.size == 5


def test_an_argument_supplied_by_the_pipeline_cannot_also_be_set_in_the_config():
    # The supplied value would win, so the config would record a setting that was never used.
    with pytest.raises(ConfigError, match="'size'.*must not be set in the config"):
        _registry_with_one_widget().build({"name": "simple", "size": 3}, size=5)


def test_unknown_name_lists_the_available_ones():
    with pytest.raises(ConfigError, match="Available: simple"):
        _registry_with_one_widget().build({"name": "missing"})


def test_misspelt_argument_is_reported_as_a_config_error():
    with pytest.raises(ConfigError, match="colour"):
        _registry_with_one_widget().build({"name": "simple", "size": 3, "colour": "blue"})


def test_registering_a_name_twice_is_rejected():
    registry = _registry_with_one_widget()

    with pytest.raises(ValueError, match="already registered"):
        registry.register("simple")(object)
