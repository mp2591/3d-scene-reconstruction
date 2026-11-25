"""The software and hardware a result was produced with.

Recorded in every checkpoint, evaluation and profile, so that a number can
always be traced back to the code and machine that produced it.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import importlib.util
import logging
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


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


PACKAGE_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
"""A normalised Python package name (PEP 503): lower case, so it never takes a capitalised label."""


def is_valid_package(name, version) -> bool:
    """Whether a name and version can stand in a software record: a normalised name, a version without spaces.

    The version rule is kept loose on purpose: a strict version grammar would
    turn real installations with unusual version strings into unknowns.
    """
    valid_name = isinstance(name, str) and PACKAGE_NAME.fullmatch(name) is not None
    return valid_name and isinstance(version, str) and re.fullmatch(r"\S+", version) is not None


def evaluation_software(device: torch.device) -> dict[str, Any]:
    """The software and device of an evaluation, saved with it so that the suite report can compare runs.

    Two evaluations count as made with the same software when these records
    are identical (see ``scene_recon.report``). That covers exactly the
    fields below. Not covered: two builds of a package that carry the same
    version number, the GPU driver, the operating system, and the source
    revision (a change of the code that alters results raises
    ``COMPATIBILITY_VERSION`` instead).

    Every installed package is listed, not only those known to matter, so a
    package a new model or metric brings along is covered without anyone
    having to name it.

    Returns:
        ``python``: its version; ``cuda``: the CUDA version PyTorch was
        built with (``"none"`` without); ``device_type``: ``cuda``, ``cpu``,
        ...; ``device``: the device's model (see ``device_model``), or
        ``None`` if it cannot be named; ``packages``: every installed
        package and its version by normalised name (``scikit-image``,
        ``torch``), or ``None`` if the metadata of a package cannot be read.
        The report never takes a record with ``None`` for the same software.
    """
    return {
        "python": platform.python_version(),
        "cuda": torch.version.cuda or "none",
        "device_type": device.type,
        "device": device_model(device),
        "packages": installed_packages(),
    }


def installed_packages() -> dict[str, str] | None:
    """Every installed Python package and its version, by normalised name, sorted.

    If a package is installed twice, the copy Python imports (the first on
    the search path) is listed. ``None``, with a warning, if the metadata
    of an installed package cannot be read or has no valid name or version
    (``is_valid_package``; the warning names the package's folder), or if
    the installed packages cannot be listed at all.
    """
    packages: dict[str, str] = {}
    try:
        distributions = iter(importlib.metadata.distributions())
    except Exception as error:
        _warn_packages_cannot_be_listed(error)
        return None
    while True:
        try:
            distribution = next(distributions, _NO_MORE_PACKAGES)
            if distribution is None:
                raise TypeError("a package finder listed None instead of a package")
        except Exception as error:  # finding the next package failed: no package is to blame
            _warn_packages_cannot_be_listed(error)
            return None
        if distribution is _NO_MORE_PACKAGES:
            return dict(sorted(packages.items()))
        try:
            name, version = distribution.metadata["Name"], distribution.version
            normalised = re.sub(r"[-_.]+", "-", name).lower() if isinstance(name, str) else name
            if not is_valid_package(normalised, version):
                raise ValueError(f"no valid name or version (name {name!r}, version {version!r})")
        except Exception as error:  # any damage: an unreadable file, not UTF-8, a broken archive, a missing field
            logger.warning(
                "The metadata of the installed package in %s cannot be read (%s: %s), so the software of this "
                "evaluation is not recorded. Reinstall or remove that package.",
                _metadata_folder(distribution),
                type(error).__name__,
                error,
            )
            return None
        packages.setdefault(normalised, version)


_NO_MORE_PACKAGES = object()
"""Marks the end of the list of installed packages (a broken package finder may list ``None``)."""


def _metadata_folder(distribution: Any) -> str:
    """Where an installed package's metadata lies, for a warning; never raises."""
    try:
        return str(getattr(distribution, "_path", "an unknown folder"))
    except Exception:
        return "an unknown folder"


def _warn_packages_cannot_be_listed(error: Exception) -> None:
    logger.warning(
        "The installed packages cannot be listed (%s: %s), so the software of this evaluation is not recorded. "
        "Check that every folder on Python's search path (sys.path) can be read.",
        type(error).__name__,
        error,
    )


def device_model(device: torch.device) -> str | None:
    """The model of the device scores are computed on, or ``None`` if it cannot be named.

    The model of an NVIDIA GPU, or of the CPU as the operating system
    reports it, followed on the CPU by the number of threads PyTorch uses
    (how a sum is split between threads can change its last digits). Other
    accelerators are not named.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device) if torch.cuda.is_available() else None
    if device.type == "cpu":
        model = cpu_model()
        return f"{model}, {torch.get_num_threads()} threads" if model else None
    return None


@functools.lru_cache(maxsize=1)
def cpu_model() -> str | None:
    """The CPU's model name as the operating system reports it, or ``None`` if it cannot be read."""
    try:
        if sys.platform == "win32":
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        elif sys.platform == "darwin":
            command = ["sysctl", "-n", "machdep.cpu.brand_string"]
            name = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True).stdout
        else:
            lines = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines()
            name = next((line.split(":", 1)[1] for line in lines if line.startswith("model name")), "")
    except (OSError, subprocess.SubprocessError):
        return None
    return " ".join(str(name).split()) or None


def device_description(device: torch.device) -> dict[str, Any]:
    """What a speed measurement depends on: the device, its model, the software and the CPU threads.

    The software is PyTorch and ``timm``, whose code runs the ConvNeXt V2 encoder.
    """
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CUDA not available"
    else:
        name = f"{platform.machine()} CPU, {torch.get_num_threads()} threads"
    return {"device": str(device), "name": name, "torch": str(torch.__version__), "timm": loaded_version("timm")}


def package_version(name: str) -> str | None:
    """The installed version of a Python package, or ``None`` if it is not installed or its metadata cannot be read."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception as error:  # damaged metadata must not stop the work whose environment is recorded
        logger.warning("The version of %s cannot be read (%s: %s).", name, type(error).__name__, error)
        return None


UNREADABLE_VERSION = "unreadable"
"""What ``loaded_version`` gives for a package that is installed but whose version cannot be read."""


def loaded_version(name: str) -> str | None:
    """The version of a package's code, for a record that decides whether a saved result is reused.

    Read from the package's metadata or, if that is damaged, from the
    ``__version__`` of the module itself, so that damaged metadata does not
    hide an update. ``None`` if the package is not installed;
    ``UNREADABLE_VERSION`` if it is installed but neither can be read.
    ``profile_status`` never reuses a speed measured while the version of
    ``timm`` is unreadable.
    """
    try:
        version = importlib.metadata.version(name)
    except Exception:  # not installed, or its metadata is damaged: the module itself is asked next
        version = None
    if isinstance(version, str) and version.strip():
        return version
    try:
        if importlib.util.find_spec(name) is None:
            return None  # not installed
        version = getattr(importlib.import_module(name), "__version__", None)
    except Exception as error:  # installed, but it cannot be imported
        logger.warning("The version of %s cannot be read (%s: %s).", name, type(error).__name__, error)
        return UNREADABLE_VERSION
    if isinstance(version, str) and version.strip():
        return version
    logger.warning("The version of %s cannot be read: neither its metadata nor the module gives one.", name)
    return UNREADABLE_VERSION


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
