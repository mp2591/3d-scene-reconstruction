"""Models. Importing this package registers every model found inside it.

Any Python file placed in this directory (or a sub-directory with an
``__init__.py``) is imported automatically, so its ``@MODELS.register``
decorators run without further wiring. To add a model: create the file,
decorate the class, and name it in a config.
"""

import importlib
import pkgutil

from scene_recon.models.base import LEVEL_COARSE, LEVEL_FINE, ParameterGroup, ReconstructionModel

__all__ = ["LEVEL_COARSE", "LEVEL_FINE", "ParameterGroup", "ReconstructionModel"]

for _module in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
    importlib.import_module(_module.name)
