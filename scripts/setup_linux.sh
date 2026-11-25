#!/usr/bin/env bash
# Prepares a Linux machine with an NVIDIA GPU (written for a GH200) to run
# the comparison. Run it from anywhere, once, after cloning the repository:
#
#   bash scripts/setup_linux.sh
#
# It is safe to run again, also after an interruption: finished steps are
# skipped, unfinished ones start over. What it does:
#   1. Creates the virtual environment .venv and installs the package with
#      PyTorch, LPIPS and pytest.
#   2. Checks that PyTorch can use the GPU.
#   3. Downloads the Hiera-T weights into weights/ (661 MB).
#   4. Downloads ShapeNet-SRN Cars into data/srn_cars/ (2.4 GB, from
#      pixelNeRF's shared Google Drive folder), unless a complete copy is
#      there already: every split with its number of scenes, every scene with
#      its intrinsics and an image and a pose for each of its views.
#   5. Runs the tests.
#
# If the download of the cars fails (Google Drive limits how often a file can
# be downloaded), copy the dataset from a machine that has it, from that
# machine:
#   rsync -a data/srn_cars/ <user>@<server>:3d-scene-reconstruction/data/srn_cars/
# and run this script again.

set -euo pipefail
cd "$(dirname "$0")/.."

log() { echo "$(date '+%F %T') | $*"; }

# ---------------------------------------------------------------- 1. environment
# A venv interrupted while it was being made lacks a working pip or its activation script: make it again.
if ! .venv/bin/python -m pip --version > /dev/null 2>&1 || [ ! -r .venv/bin/activate ]; then
    rm -rf .venv
    log "Creating the virtual environment .venv"
    if ! python3 -m venv .venv; then
        echo "python3 cannot create a virtual environment. Install its venv module, e.g." >&2
        echo "  sudo apt install python3-venv" >&2
        echo "then run this script again." >&2
        rm -rf .venv
        exit 1
    fi
    .venv/bin/python -m pip install --upgrade pip
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# Running this script again installs nothing new: the software that scores the runs stays the same.
log "Installing the package and its dependencies"
# Since PyTorch 2.11 the package on PyPI includes CUDA on ARM machines such as the GH200 too.
python -m pip install -e ".[lpips,dev]"

# ---------------------------------------------------------------- 2. GPU
log "Checking that PyTorch can use the GPU"
if ! python - <<'EOF'
import sys

import torch

if not torch.cuda.is_available():
    sys.exit(
        f"PyTorch {torch.__version__} (built for CUDA {torch.version.cuda}) cannot use the GPU.\n"
        "If nvidia-smi works, the driver is probably older than that CUDA version. Install a PyTorch\n"
        "built for an older CUDA, then run this script again, e.g.:\n"
        "  .venv/bin/python -m pip install --force-reinstall torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu128"
    )
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
EOF
then
    nvidia-smi || true
    exit 1
fi

# ---------------------------------------------------------------- 3. Hiera-T weights
hiera=weights/hiera_tiny_224_mae_in1k.pth
if [ ! -f "$hiera" ]; then
    log "Downloading the Hiera-T weights (661 MB) to $hiera"
    mkdir -p weights
    curl --fail -L -o weights/download.part https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth
    mv weights/download.part "$hiera"
fi

# ---------------------------------------------------------------- 4. ShapeNet-SRN Cars
# Succeeds if the folder holds the complete dataset; otherwise prints what is missing. A scene is
# a folder with an intrinsics.txt, as for the loader; each has its views 000000, 000001, ...
# with an image and a pose each: 50 views in the training split, 251 in the others.
srn_complete() {
    python - "$1" <<'EOF'
import sys
from pathlib import Path

root = Path(sys.argv[1])
problems = []
for split, (expected, views) in {"cars_train": (2458, 50), "cars_val": (352, 251), "cars_test": (704, 251)}.items():
    directory = root / split
    scenes = [path for path in directory.iterdir() if (path / "intrinsics.txt").is_file()] if directory.is_dir() else []
    if len(scenes) != expected:
        problems.append(f"{directory} has {len(scenes)} scenes, expected {expected}")
        continue
    names = {f"{view:06d}" for view in range(views)}
    for scene in scenes:
        images = {path.stem for path in (scene / "rgb").glob("*.png")}
        poses = {path.stem for path in (scene / "pose").glob("*.txt")}
        if images != names or poses != names:
            problems.append(f"{scene} does not have an image and a pose for each of its {views} views")
            break
if problems:
    sys.exit("\n".join(problems))
EOF
}

# Succeeds if the file is a complete, undamaged ZIP archive.
zip_intact() {
    python - "$1" <<'EOF'
import sys
import zipfile

try:
    damaged = zipfile.ZipFile(sys.argv[1]).testzip()
except (OSError, zipfile.BadZipFile) as error:
    sys.exit(f"{sys.argv[1]} is not a ZIP archive ({error})")
if damaged is not None:
    sys.exit(f"{sys.argv[1]} is damaged (first damaged file: {damaged})")
EOF
}

if ! srn_complete data/srn_cars 2> /dev/null; then
    # Unpacked beside the final folder, which appears only once the dataset is complete.
    if [ ! -f data/srn_cars.zip ]; then
        log "Downloading ShapeNet-SRN Cars (2.4 GB)"
        mkdir -p data
        curl --fail -L -o data/srn_cars.zip.part \
            "https://drive.usercontent.google.com/download?id=19yDsEJjx9zNpOKz9o6AaK-E8ED6taJWU&export=download&confirm=t"
        # Google Drive answers with a web page instead of the file when it refuses the download.
        if ! zip_intact data/srn_cars.zip.part; then
            rm data/srn_cars.zip.part
            echo "The download is not the dataset (Google Drive may have refused it). Run this script again" >&2
            echo "later, or copy the dataset from another machine (see the top of this script)." >&2
            exit 1
        fi
        mv data/srn_cars.zip.part data/srn_cars.zip
    fi
    log "Unpacking it"
    rm -rf data/srn_cars.partial
    python -m zipfile -e data/srn_cars.zip data/srn_cars.partial
    if ! srn_complete data/srn_cars.partial; then
        echo "The downloaded dataset is incomplete. Delete data/srn_cars.zip and run this script again," >&2
        echo "or copy the dataset from another machine (see the top of this script)." >&2
        exit 1
    fi
    if [ -e data/srn_cars ]; then
        # Kept, not deleted: it may hold files of its own.
        aside="data/srn_cars.incomplete.$(date +%Y%m%d-%H%M%S)"
        log "Moving the incomplete data/srn_cars aside to $aside"
        mv data/srn_cars "$aside"
    fi
    mv data/srn_cars.partial data/srn_cars
    rm data/srn_cars.zip
fi
log "ShapeNet-SRN Cars is complete: 2458 training, 352 validation and 704 test scenes"

# ---------------------------------------------------------------- 5. tests
log "Running the tests (a few minutes)"
python -m pytest -q

log "Ready. Start the comparison inside tmux with: bash scripts/run_comparison.sh runs"
log "(see docs/running_on_a_gpu_server.md)"
