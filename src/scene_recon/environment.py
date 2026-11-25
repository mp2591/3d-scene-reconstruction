"""The software and hardware a result was produced with.

Recorded in every checkpoint, evaluation and profile, so that a number can
always be traced back to the code and machine that produced it.
"""

from __future__ import annotations

import functools
import importlib.metadata
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import torch


@functools.lru_cache(maxsize=1)
def environment() -> dict[str, Any]:
    """Versions, source revision and hardware of the current process."""
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),  # a str subclass that restricted loading would reject
        "timm": package_version("timm"),  # implements the ConvNeXt V2 encoder
        "lpips": package_version("lpips"),  # implements the LPIPS metric
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        **_source_revision(),
    }


def evaluation_software(device: torch.device) -> dict[str, Any]:
    """What a score can depend on besides the model and the data, saved with every evaluation.

    Every installed package is listed, not only those known to matter, so a
    package a new model or metric brings along is covered without anyone
    having to name it. The suite report compares these between runs (see
    ``scene_recon.report``). The source revision is left out: a change that
    alters results raises ``COMPATIBILITY_VERSION`` instead.

    Returns:
        ``python``: its version; ``cuda``: the CUDA version PyTorch was
        built with (``"none"`` without); ``device``: the name of the device
        the scores were computed on (floating-point results can differ
        slightly between GPU models); ``packages``: every installed package
        and its version, by normalised name (``scikit-image``, ``torch``).
    """
    return {
        "python": platform.python_version(),
        "cuda": torch.version.cuda or "none",
        "device": device_description(device)["name"],
        "packages": installed_packages(),
    }


def installed_packages() -> dict[str, str]:
    """Every installed Python package and its version, by normalised name, sorted.

    If a package is installed twice, the copy Python imports (the first on
    the search path) is listed.
    """
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            packages.setdefault(re.sub(r"[-_.]+", "-", name).lower(), distribution.version)
    return dict(sorted(packages.items()))


def device_description(device: torch.device) -> dict[str, Any]:
    """What a speed measurement depends on: the device, its model, the software and the CPU threads.

    The software is PyTorch and ``timm``, whose code runs the ConvNeXt V2 encoder.
    """
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CUDA not available"
    else:
        name = f"{platform.machine()} CPU, {torch.get_num_threads()} threads"
    return {"device": str(device), "name": name, "torch": str(torch.__version__), "timm": package_version("timm")}


def package_version(name: str) -> str | None:
    """The installed version of a Python package, or ``None`` if it is not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_revision() -> dict[str, Any]:
    """The git commit of this code base, and whether it had uncommitted changes."""
    package_dir = Path(__file__).parent

    def git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=package_dir, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None  # not a git checkout, or git is not installed
        return result.stdout.strip()

    revision = git("rev-parse", "HEAD")
    if revision is None:
        return {"source_revision": None, "source_modified": None}
    return {"source_revision": revision, "source_modified": bool(git("status", "--porcelain"))}
