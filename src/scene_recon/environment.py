"""The software and hardware a result was produced with.

Recorded in every checkpoint, evaluation and profile, so that a number can
always be traced back to the code and machine that produced it.
"""

from __future__ import annotations

import functools
import platform
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
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        **_source_revision(),
    }


def device_description(device: torch.device) -> dict[str, Any]:
    """What a speed measurement depends on: the device, its model, the software and the CPU threads."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CUDA not available"
    else:
        name = f"{platform.machine()} CPU, {torch.get_num_threads()} threads"
    return {"device": str(device), "name": name, "torch": str(torch.__version__)}


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
