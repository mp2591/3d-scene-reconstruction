"""Name-based registries that connect YAML configs to Python classes.

A component (a model, a dataset, a loss, ...) is registered under a short
name with a decorator::

    @MODELS.register("my_model")
    class MyModel(ReconstructionModel):
        def __init__(self, hidden_width: int = 256): ...

and selected in a config by that name. Every other key in the config
section is passed to the constructor as a keyword argument::

    model:
      name: my_model
      hidden_width: 512
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping


class ConfigError(ValueError):
    """Raised when a config file asks for something the pipeline cannot build."""


class Registry:
    """A lookup table from a short name to the class that implements it."""

    def __init__(self, kind: str) -> None:
        """
        Args:
            kind: What this registry holds, e.g. ``"model"``. Only used to
                make error messages readable.
        """
        self.kind = kind
        self._classes: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a class decorator that registers the class under ``name``."""

        def decorator(cls: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._classes:
                raise ValueError(
                    f"A {self.kind} named '{name}' is already registered "
                    f"({self._classes[name].__qualname__})."
                )
            self._classes[name] = cls
            return cls

        return decorator

    def names(self) -> list[str]:
        """All registered names, sorted."""
        return sorted(self._classes)

    def get(self, name: str) -> Callable[..., Any]:
        """Return the class registered under ``name``."""
        if name not in self._classes:
            raise ConfigError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or '(none)'}."
            )
        return self._classes[name]

    def build(self, config: Mapping[str, Any], **extra_arguments: Any) -> Any:
        """Instantiate the component described by a config section.

        Args:
            config: A mapping with a ``name`` key. All other keys become
                constructor keyword arguments.
            extra_arguments: Additional constructor arguments supplied by
                the pipeline rather than by the config file.
        """
        if "name" not in config:
            raise ConfigError(f"A {self.kind} config needs a 'name' key, got: {dict(config)}.")

        arguments = {key: value for key, value in config.items() if key != "name"}
        arguments.update(extra_arguments)
        cls = self.get(config["name"])

        # Check the arguments against the constructor signature first, so a
        # typo in a config key is reported as a config problem rather than
        # as a TypeError from deep inside the constructor.
        try:
            inspect.signature(cls).bind(**arguments)
        except TypeError as error:
            raise ConfigError(
                f"Cannot build {self.kind} '{config['name']}' ({cls.__qualname__}): {error}. "
                f"Its constructor is {cls.__name__}{inspect.signature(cls)}."
            ) from error

        return cls(**arguments)


# The pipeline itself only ever builds from these four...
MODELS = Registry("model")
DATASETS = Registry("dataset")
LOSSES = Registry("loss")
METRICS = Registry("metric")

# ...while these two exist for model authors. A top-level model is free to
# build its sub-networks however it likes; registering encoders and decoders
# here lets a config mix and match them (see models/testing/tiny_model.py).
ENCODERS = Registry("encoder")
DECODERS = Registry("decoder")
