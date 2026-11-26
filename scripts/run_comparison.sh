#!/usr/bin/env bash
# Runs the whole comparison of the proposal from start to finish, without
# anyone having to step in (docs/running_on_a_gpu_server.md). Prepare the
# machine with scripts/setup_linux.sh first, then, inside tmux:
#
#   bash scripts/run_comparison.sh runs [STEPS_PER_RUN]
#
# "runs" is the output folder (relative to the repository, or absolute).
# STEPS_PER_RUN is the step budget: how many steps each comparison run
# trains, over which its learning rate falls tenfold (optimization.decay_steps).
# Without it, the budget of configs/suites/proposal_comparison.yaml is used
# (100,000). The pilot shares the learning-rate schedule, so the budget can
# change only until a pilot run has saved a checkpoint; it is kept in
# <output>/steps_per_run, so that the same command without it carries on
# with the same budget.
#
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
#      Then, once, an estimate of how long the rest takes on this machine for
#      a few budgets (scene-recon estimate-time), kept in
#      <output>/time_estimate.txt. To change the budget, stop the script
#      (Ctrl-c) and start it again with another STEPS_PER_RUN before the
#      first pilot run has saved its first checkpoint.
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
# afterwards, so the repository records the rates that were used (and, with
# another budget, set it in both suite files, as the script says at the end).

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

# The step budget (see the top of this file): asked for, else the one this folder was started
# with, else the one in the comparison's suite file.
budget_file="$output/steps_per_run"
suite_steps=$(python -c 'import sys; from scene_recon.suite import load_suite
print(next(iter(load_suite(sys.argv[1]).runs.values())).training.num_steps)' "$comparison")
recorded_steps=$(cat "$budget_file" 2> /dev/null || true)
started_steps=${recorded_steps:-$suite_steps}
steps=${asked_steps:-$started_steps}
# A run's checkpoint is <run folder>/checkpoints/last.pt (src/scene_recon/run_files.py).
pilot_checkpoint=$(find "$output/encoder_lr_pilot" -path '*/checkpoints/last.pt' -print -quit 2> /dev/null || true)
if [ -n "$pilot_checkpoint" ] && [ "$steps" != "$started_steps" ]; then
    echo "The pilot has trained with a budget of $started_steps steps per run ($pilot_checkpoint)," >&2
    echo "and its learning-rate schedule cannot change any more. Start again with that budget, or without one:" >&2
    echo "  bash scripts/run_comparison.sh $output" >&2
    echo "To use $steps steps instead, every pilot and comparison run must train again from the start:" >&2
    echo "delete $output/encoder_lr_pilot and $output/proposal_comparison first." >&2
    exit 1
fi
# The pilot shares the comparison's learning-rate schedule, but trains its own number of steps.
both=(optimization.decay_steps="$steps")
comparison_only=(training.num_steps="$steps")

log "Step 0: checking the GPU and that the pilot fits the comparison ($steps training steps per run)"
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
scene-recon choose-encoder-rates --pilot "$pilot" --comparison "$comparison" --check-only \
    --set "${settings[@]}" "${both[@]}" --comparison-set "${comparison_only[@]}"
echo "$steps" > "$budget_file"  # kept once it has passed the check, e.g. that it is not shorter than the pilot

# The trial uses the rates of the comparison file as it was before step 3,
# so it is not run again once it has passed. Its own short length comes last, so it wins.
trial_done="$output/trial/passed"
if [ ! -e "$trial_done" ]; then
    log "Step 1: a short trial of every run in $output/trial"
    scene-recon run-suite --config "$comparison" --set "${settings[@]}" "${both[@]}" "${comparison_only[@]}" \
        experiment.output_dir="$output/trial" "${trial[@]}"
    touch "$trial_done"
fi

# Made once; a failed estimate is reported and the runs go on without it.
estimate="$output/time_estimate.txt"
if [ ! -e "$estimate" ]; then
    log "Step 1: timing every run on this machine, to estimate how long the rest takes"
    budgets=$(printf '%s\n' "$steps" 100000 150000 200000 300000 | sort -n -u)
    # shellcheck disable=SC2086  # $budgets is a list of numbers
    if scene-recon estimate-time --pilot "$pilot" --comparison "$comparison" --steps $budgets \
            --first-scenes "$first_scenes" --set "${settings[@]}" "${both[@]}" \
            --comparison-set "${comparison_only[@]}" > "$estimate.partial"; then
        mv "$estimate.partial" "$estimate"
    else
        echo "The time estimate failed (see above); carrying on without it." >&2
    fi
fi
if [ -e "$estimate" ]; then
    cat "$estimate"
fi
echo "This folder trains $steps steps per comparison run."
if [ -z "$pilot_checkpoint" ]; then
    echo "For another budget, stop the script now (Ctrl-c) and start it again with that number, e.g."
    echo "  bash scripts/run_comparison.sh $output 200000"
    echo "That works until the first pilot run saves its first checkpoint (training.checkpoint_every steps)."
fi

log "Step 2: the learning-rate pilot"
scene-recon run-suite --config "$pilot" --set "${settings[@]}" "${both[@]}"

log "Step 3: choosing each encoder's learning rate"
scene-recon choose-encoder-rates --pilot "$pilot" --comparison "$comparison" \
    --set "${settings[@]}" "${both[@]}" --comparison-set "${comparison_only[@]}"

# The comparison's settings, from here on.
comparison_settings=("${settings[@]}" "${both[@]}" "${comparison_only[@]}")

first_report="$output/proposal_comparison_first_${first_scenes}_test_scenes"
if [ ! -e "$first_report" ]; then
    log "Step 4: training the comparison, scoring it on the first $first_scenes test scenes"
    scene-recon run-suite --config "$comparison" --set "${comparison_settings[@]}" evaluation.num_scenes="$first_scenes"
    # Copied beside its final place, which appears only once the copy is complete.
    rm -rf "$first_report.partial"
    mkdir "$first_report.partial"
    cp -r "$output/proposal_comparison/"{summary.md,summary.csv,paired_comparison.csv,plots} "$first_report.partial/"
    mv "$first_report.partial" "$first_report"
fi

log "Step 5: scoring the comparison on every test scene"
scene-recon run-suite --config "$comparison" --set "${comparison_settings[@]}"

log "Finished. Report: $output/proposal_comparison/summary.md"
if [ "$steps" != "$suite_steps" ]; then
    echo "The runs trained $steps steps, not the $suite_steps of the suite files. When you commit the chosen rates,"
    echo "also set training.num_steps and optimization.decay_steps to $steps in $comparison,"
    echo "and optimization.decay_steps to $steps in $pilot, so that the repository records the budget."
fi
