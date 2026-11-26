#!/usr/bin/env bash
# Runs the whole comparison of the proposal from start to finish, without
# anyone having to step in (docs/running_on_a_gpu_server.md). Prepare the
# machine with scripts/setup_linux.sh first, then, inside tmux:
#
#   bash scripts/run_comparison.sh runs
#
# "runs" is the output folder (relative to the repository, or absolute).
# Everything the script prints also goes to <output>/run_comparison.log.
# Every step skips the work it has already done, so after a crash or a
# reboot the same command carries on where it stopped. The steps:
#
#   0. Check that the GPU works and that the pilot fits the comparison
#      (seconds). Every run is told to use the GPU, so none can fall back to
#      the processor unnoticed. Also prints the GPU memory, processors and
#      shared memory this machine gives the runs, and warns if the repository
#      or the output folder is on a container's own disk.
#   1. A short trial of every comparison run, in <output>/trial, so that a
#      problem with this machine shows up within minutes, not after hours.
#   2. The learning-rate pilot (configs/suites/encoder_lr_pilot.yaml).
#   3. Choose each encoder's rate, write it into
#      configs/suites/proposal_comparison.yaml and copy the chosen pilot runs
#      into the comparison's folder (scene-recon choose-encoder-rates).
#   4. The comparison: training to the end, scores on the first 100 test
#      scenes, speed. A copy of that report is kept in
#      <output>/proposal_comparison_first_100_test_scenes/. Once that copy
#      exists, this step is not run again (it would undo step 5's scores):
#      after changing the comparison, delete the copy to make it again.
#   5. The comparison scored on every test scene: the final report is
#      <output>/proposal_comparison/summary.md.
#
# Step 3 changes configs/suites/proposal_comparison.yaml; commit that change
# afterwards, so the repository records the rates that were used.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -ne 1 ]; then
    echo "Usage: bash scripts/run_comparison.sh OUTPUT_FOLDER" >&2
    exit 2
fi
output=$1
pilot=configs/suites/encoder_lr_pilot.yaml
comparison=configs/suites/proposal_comparison.yaml
first_scenes=100   # test scenes scored first, so that every run has results early

# Settings given to every command: where the results go, and the GPU.
settings=(experiment.output_dir="$output" experiment.device=cuda)

# A few steps of every run, with every stage, on a few scenes.
trial=(training.num_steps=50 training.validate_every=50 training.checkpoint_every=50
       evaluation.num_scenes=2 evaluation.num_target_views=2)

mkdir -p "$output"
exec > >(tee -a "$output/run_comparison.log") 2>&1

log() { echo "$(date '+%F %T') | $*"; }

# A container's own disk (overlay) is deleted when the container stops or restarts, with every
# checkpoint on it. Warns about each given folder that is on such a disk.
warn_if_container_disk() {
    for folder in "$@"; do
        if [ "$(stat -f -c %T "$folder" 2> /dev/null)" = overlayfs ]; then
            echo "WARNING: $(cd "$folder" && pwd) is on the container's own disk, which is deleted when the" >&2
            echo "container stops or restarts. Put the repository and the output folder on a persistent volume" >&2
            echo "(docs/running_on_a_gpu_server.md, 'On a Run:ai cluster')." >&2
        fi
    done
}

if [ ! -x .venv/bin/scene-recon ]; then
    echo "The environment .venv is missing or incomplete: run bash scripts/setup_linux.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# Pretrained weights downloaded by the runs (ConvNeXt V2 from Hugging Face, the VGG16 of LPIPS)
# are kept in weights/ with the repository, not in the home folder, which a container may lose.
export HF_HOME="${HF_HOME:-$PWD/weights/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/weights/torch}"

log "Step 0: checking the GPU and that the pilot fits the comparison"
warn_if_container_disk . "$output"
# Also prints what this machine gives the runs. In a container with a share of a GPU (Run:ai)
# the memory is that share; the runs needed at most 22 GB on an RTX 4090.
python - <<'EOF'
import os
import shutil
import sys
from pathlib import Path

import torch

try:
    torch.ones(1, device="cuda").sum().item()
except Exception as error:
    sys.exit(
        f"PyTorch cannot use the GPU ({error}).\n"
        "Check nvidia-smi; after a driver change, run bash scripts/setup_linux.sh again."
    )
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}, {total / 1e9:.1f} GB of memory, {free / 1e9:.1f} GB of it free")
# The rest is for information only: what cannot be read is reported as such.
try:
    cpus = f"{len(os.sched_getaffinity(0))} processor threads this process may run on"
except (AttributeError, OSError):  # no affinity on this system
    cpus = f"{os.cpu_count()} processor threads in the machine"
try:  # a container's own limit (cgroup v2): "max" or "<quota> <period>"; a parent's may be lower
    quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    if quota != "max":
        cpus += f", of which the container may keep {int(quota) / int(period):g} busy"
except (OSError, ValueError):
    pass
print(f"CPUs: {cpus}")
try:
    shared = shutil.disk_usage("/dev/shm")
    print(f"Shared memory (/dev/shm): {shared.total / 1e6:.0f} MB, {shared.free / 1e6:.0f} MB of it free")
except OSError as error:
    print(f"Shared memory (/dev/shm): cannot be read ({error})")
# A cache folder the container image sets (HF_HUB_CACHE, TORCH_HOME) wins over weights/ set above.
try:
    from huggingface_hub import constants

    print(f"Downloaded weights go to: {constants.HF_HUB_CACHE} (Hugging Face), {torch.hub.get_dir()} (PyTorch)")
except Exception as error:
    print(f"Downloaded weights go to: cannot be read ({error})")
EOF
scene-recon choose-encoder-rates --pilot "$pilot" --comparison "$comparison" --check-only --set "${settings[@]}"

# The trial uses the rates of the comparison file as it was before step 3,
# so it is not run again once it has passed.
trial_done="$output/trial/passed"
if [ ! -e "$trial_done" ]; then
    log "Step 1: a short trial of every run in $output/trial"
    scene-recon run-suite --config "$comparison" --set "${settings[@]}" \
        experiment.output_dir="$output/trial" "${trial[@]}"
    touch "$trial_done"
fi

log "Step 2: the learning-rate pilot"
scene-recon run-suite --config "$pilot" --set "${settings[@]}"

log "Step 3: choosing each encoder's learning rate"
scene-recon choose-encoder-rates --pilot "$pilot" --comparison "$comparison" --set "${settings[@]}"

first_report="$output/proposal_comparison_first_${first_scenes}_test_scenes"
if [ ! -e "$first_report" ]; then
    log "Step 4: training the comparison, scoring it on the first $first_scenes test scenes"
    scene-recon run-suite --config "$comparison" --set "${settings[@]}" evaluation.num_scenes="$first_scenes"
    # Copied beside its final place, which appears only once the copy is complete.
    rm -rf "$first_report.partial"
    mkdir "$first_report.partial"
    cp -r "$output/proposal_comparison/"{summary.md,summary.csv,paired_comparison.csv,plots} "$first_report.partial/"
    mv "$first_report.partial" "$first_report"
fi

log "Step 5: scoring the comparison on every test scene"
scene-recon run-suite --config "$comparison" --set "${settings[@]}"

log "Finished. Report: $output/proposal_comparison/summary.md"
