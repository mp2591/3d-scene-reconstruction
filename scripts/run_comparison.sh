#!/usr/bin/env bash
# Runs the whole comparison of the proposal from start to finish, without
# anyone having to step in (docs/running_on_a_gpu_server.md). Prepare the
# machine with scripts/setup_linux.sh first, then, inside tmux:
#
#   bash scripts/run_comparison.sh runs [STEPS_PER_RUN]
#
# "runs" is the output folder (relative to the repository, or absolute).
# STEPS_PER_RUN is the step budget: how many steps each run trains. Without
# it, the budget of configs/suites/proposal_comparison.yaml is used (400,000,
# pixelNeRF's). With it, three settings follow the budget: training.num_steps,
# optimization.decay_steps (the learning rate falls tenfold over the budget)
# and training.first_phase_steps (three quarters of it, rounded up, as pixelNeRF's
# 300,000 of 400,000). The learning-rate schedule is fixed once a run has
# trained, so the budget can change only until the first comparison run has
# saved a checkpoint. It is kept in <output>/steps_per_run, so that the same
# command without it carries on with the same budget.
#
# Everything the script prints also goes to <output>/run_comparison.log.
# Every step skips the work it has already done, so after a crash or a
# reboot the same command carries on where it stopped. The steps:
#
#   0. Check that the GPU works (seconds). Every run is told to use the GPU,
#      so none can fall back to the processor unnoticed. Also prints the GPU
#      memory, processors and shared memory this machine gives the runs, and
#      warns if the repository or the output folder is on a container's own disk.
#   1. A short trial of every run, in <output>/trial, so that a problem with
#      this machine shows up within minutes, not after hours. Then, once, an
#      estimate of how long the rest takes on this machine for a few budgets
#      (scene-recon estimate-time), kept in <output>/time_estimate.txt. To
#      change the budget, stop the script (Ctrl-c) and start it again with
#      another STEPS_PER_RUN before the first run has saved its first checkpoint.
#   2. The comparison: training to the end, scores on the first 100 test
#      scenes, speed. A copy of that report is kept in
#      <output>/proposal_comparison_first_100_test_scenes/. Once that copy
#      exists, this step is not run again (it would undo step 3's scores):
#      after changing the comparison, delete the copy to make it again.
#   3. The comparison scored on every test scene: the final report is
#      <output>/proposal_comparison/summary.md.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: bash scripts/run_comparison.sh OUTPUT_FOLDER [STEPS_PER_RUN]" >&2
    exit 2
fi
output=$1
asked_steps=${2:-}
if [ -n "$asked_steps" ] && ! [[ $asked_steps =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS_PER_RUN must be a whole number of training steps, e.g. 200000." >&2
    exit 2
fi
comparison=configs/suites/proposal_comparison.yaml
first_scenes=100   # test scenes scored first, so that every run has results early

# Settings given to every command: where the results go, and the GPU.
settings=(experiment.output_dir="$output" experiment.device=cuda)

# A few steps of every run, with every stage, on a few scenes. Half of them in
# the first phase (two source views, rays from the object), half after it.
trial=(training.num_steps=50 training.first_phase_steps=25 training.validate_every=50
       training.checkpoint_every=50 evaluation.num_scenes=2 evaluation.num_target_views=2)

mkdir -p "$output"
# One launch per output folder at a time: two would train the same runs into the same files.
# The training processes inherit the lock, so it is held until the last of them has stopped.
if command -v flock > /dev/null; then
    exec 9> "$output/.run_comparison.lock"
    if ! flock -n 9; then
        echo "Another run_comparison.sh, or a training process it started, is still using $output." >&2
        echo "Stop it first (see docs/running_on_a_gpu_server.md, 'Start the comparison')." >&2
        exit 1
    fi
fi
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

# The step budget (see the top of this file): asked for, else the one this folder was started
# with, else the one in the comparison's suite file.
budget_file="$output/steps_per_run"
suite_steps=$(python -c 'import sys; from scene_recon.suite import load_suite
print(next(iter(load_suite(sys.argv[1]).runs.values())).training.num_steps)' "$comparison")
file_steps=$(cat "$budget_file" 2> /dev/null || true)
if ! [[ $file_steps =~ ^[1-9][0-9]*$ ]]; then file_steps=""; fi  # missing or damaged
recorded_steps=$file_steps
# A run's checkpoint is <run folder>/checkpoints/last.pt (src/scene_recon/run_files.py).
checkpoint=$(find "$output/proposal_comparison" -path '*/checkpoints/last.pt' -print -quit 2> /dev/null || true)
if [ -z "$recorded_steps" ] && [ -n "$checkpoint" ]; then
    # The record is lost, but the checkpoint says which schedule its run trained with.
    recorded_steps=$(python -c 'import sys; from pathlib import Path
from scene_recon.checkpoints import read_checkpoint_config
print(read_checkpoint_config(Path(sys.argv[1]))["optimization"]["decay_steps"])' "$checkpoint")
fi
started_steps=${recorded_steps:-$suite_steps}
steps=${asked_steps:-$started_steps}
if [ -n "$checkpoint" ] && [ "$steps" != "$started_steps" ]; then
    echo "The comparison has trained with a budget of $started_steps steps per run ($checkpoint)," >&2
    echo "and its learning-rate schedule cannot change any more. Start again with that budget, or without one:" >&2
    echo "  bash scripts/run_comparison.sh $output" >&2
    echo "To use $steps steps instead, start in a new output folder, where everything trains from the start:" >&2
    echo "  bash scripts/run_comparison.sh ${output%/}_$steps $steps" >&2
    exit 1
fi
# Written whole or not at all, so that a stop in the middle cannot leave an empty record.
if [ "$steps" != "$file_steps" ]; then
    echo "$steps" > "$budget_file.partial"
    mv "$budget_file.partial" "$budget_file"
fi
# The settings that follow the budget (see the top of this file).
first_phase=$(( (steps * 3 + 3) / 4 ))   # rounded up, so that even a one-step trial has a first phase
budget=(training.num_steps="$steps" optimization.decay_steps="$steps" training.first_phase_steps="$first_phase")

# A run trained further than the budget (by hand, see "Time to spare at the end" in
# docs/running_on_a_gpu_server.md) would be scored, but left out of the comparison.
python - "$output/proposal_comparison" "$steps" <<'EOF'
import sys
from pathlib import Path

from scene_recon.checkpoints import read_progress

folder, steps = Path(sys.argv[1]), int(sys.argv[2])
longer = {}
for checkpoint in sorted(folder.glob("*/*/checkpoints/last.pt")):  # <run>/<seed>/checkpoints/last.pt
    progress = read_progress(checkpoint)
    if progress is not None and progress.step > steps:
        longer[checkpoint.parent.parent.relative_to(folder).as_posix()] = progress.step
if longer:
    runs = ", ".join(f"{run} ({step:,} steps)" for run, step in longer.items())
    sys.exit(
        f"Trained past this folder's budget of {steps:,} steps: {runs}. Scored now, they would be left out of the "
        f"comparison. Train every run to {max(longer.values()):,} steps by hand instead "
        "(docs/running_on_a_gpu_server.md, 'Time to spare at the end')."
    )
EOF

log "Step 0: checking the GPU ($steps training steps per run)"
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

# Not run again once it has passed. It trains with its own few steps, not the budget.
trial_done="$output/trial/passed"
if [ ! -e "$trial_done" ]; then
    log "Step 1: a short trial of every run in $output/trial"
    scene-recon run-suite --config "$comparison" --set "${settings[@]}" \
        experiment.output_dir="$output/trial" "${trial[@]}"
    touch "$trial_done"
fi

# Made once; a failed estimate is reported and the runs go on without it.
estimate="$output/time_estimate.txt"
if [ ! -e "$estimate" ]; then
    log "Step 1: timing every run on this machine, to estimate how long the rest takes"
    budgets=$(printf '%s\n' "$steps" 100000 200000 300000 400000 | sort -n -u)
    # shellcheck disable=SC2086  # $budgets is a list of numbers
    if scene-recon estimate-time --config "$comparison" --steps $budgets --first-scenes "$first_scenes" \
            --set "${settings[@]}" "${budget[@]}" > "$estimate.partial"; then
        mv "$estimate.partial" "$estimate"
    else
        echo "The time estimate failed (see above); carrying on without it." >&2
    fi
fi
if [ -e "$estimate" ]; then
    cat "$estimate"
fi
echo "This folder trains $steps steps per run."
if [ -z "$checkpoint" ]; then
    echo "For another budget, stop the script now (Ctrl-c) and start it again with that number, e.g."
    echo "  bash scripts/run_comparison.sh $output 200000"
    echo "That works until the first run saves its first checkpoint (training.checkpoint_every steps)."
fi

# The comparison's settings, from here on.
comparison_settings=("${settings[@]}" "${budget[@]}")

first_report="$output/proposal_comparison_first_${first_scenes}_test_scenes"
if [ ! -e "$first_report" ]; then
    log "Step 2: training the comparison, scoring it on the first $first_scenes test scenes"
    scene-recon run-suite --config "$comparison" --set "${comparison_settings[@]}" evaluation.num_scenes="$first_scenes"
    # Copied beside its final place, which appears only once the copy is complete.
    rm -rf "$first_report.partial"
    mkdir "$first_report.partial"
    cp -r "$output/proposal_comparison/"{summary.md,summary.csv,paired_comparison.csv,plots} "$first_report.partial/"
    mv "$first_report.partial" "$first_report"
fi

log "Step 3: scoring the comparison on every test scene"
scene-recon run-suite --config "$comparison" --set "${comparison_settings[@]}"

log "Finished. Report: $output/proposal_comparison/summary.md"
if [ "$steps" != "$suite_steps" ]; then
    echo "The runs trained $steps steps, not the $suite_steps of $comparison. To record the budget in the"
    echo "repository, add to the common block of $comparison: training.num_steps $steps,"
    echo "training.first_phase_steps $first_phase and optimization.decay_steps $steps."
fi
