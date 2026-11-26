# Starting the training

How to start the full comparison on a Linux GPU machine (for example the
Run:ai GH200), and what to keep in mind while it runs. Every detail is in
[`docs/running_on_a_gpu_server.md`](docs/running_on_a_gpu_server.md).

## What it trains

Three models, one after another, each with pixelNeRF's recipe for
ShapeNet-SRN Cars (`configs/suites/proposal_comparison.yaml`):

| Run | Encoder | Decoder |
|---|---|---|
| `convnextv2_mlp` | ConvNeXt V2-T | pixelNeRF's MLP |
| `hiera_mlp` | Hiera-T | pixelNeRF's MLP |
| `hiera_nerf` | Hiera-T | HieraNeRF bridge and attention decoder |

Each run trains 400,000 steps and is then scored with 1 and 2 source views
on every view of the 704 test cars, as pixelNeRF's benchmark does. The two
baselines, `pixelnerf_resnet34` and `transnerf_unet`, are commented out
until their model files exist.

**How long.** On an RTX 4090 in full float32: about 91 hours until every
run is trained and scored on the first 100 test cars, about 368 hours until
the final report on all 704. Scoring is most of it. A GH200 should be
faster. The real figure for your machine is printed about 15 minutes after
the start.

## Before you start

- [ ] **A new output folder.** Runs trained with older code or another
      recipe cannot be continued. Use a folder that does not exist yet.
- [ ] **Disk.** At least 20 GB free on the volume that holds the
      repository: the environment, the data (2.4 GB download), the weights
      and the checkpoints (about 0.75 GB for the three runs).
- [ ] **Internet.** The machine downloads the data, the Hiera-T weights,
      the ConvNeXt V2 weights (Hugging Face) and the VGG16 that LPIPS uses.

On a Run:ai cluster, when you submit the workload:

- [ ] **Persistent volume.** Clone the repository onto a volume that
      outlives the workload (a persistent volume claim), not the
      container's own disk and not a temporary `emptyDir`. Everything,
      checkpoints included, lives inside the repository folder.
- [ ] **Large shared memory.** Turn on `--large-shm` ("Increase shared
      memory size"). Training needs about 360 MB of `/dev/shm`; the default
      64 MB is too little, and training refuses to start.
- [ ] **At least 10 processors**, and generous main memory.
- [ ] **A fixed GPU share** (for example 0.99) that stays the same until
      the end, with nothing else scheduled on that GPU. The speeds and
      training hours are only comparable when every run had the same GPU.

## 1. Prepare the machine (about 20 minutes)

```bash
git clone https://github.com/mp2591/3d-scene-reconstruction.git
cd 3d-scene-reconstruction
bash scripts/setup_linux.sh
```

It creates `.venv`, installs everything, checks the GPU, downloads the data
and the Hiera-T weights, checks the data and runs the tests. If it stops, it
says why; run it again after fixing that. If Google Drive refuses the data
download, copy `data/srn_cars/` from a machine that has it (the command is
in the docs) and run the script again.

## 2. Start the comparison

Inside `tmux`, so that it keeps running when the connection drops:

```bash
tmux new -s comparison
bash scripts/run_comparison.sh runs
```

Leave tmux with `Ctrl-b` then `d`, come back with `tmux attach -t comparison`.
Without tmux:

```bash
nohup bash scripts/run_comparison.sh runs > /dev/null 2>&1 &
```

`runs` is the output folder; use a new name if `runs` already holds an old
comparison. Everything is also written to `runs/run_comparison.log`.

## 3. In the first 15 minutes

The script first checks the GPU and prints what the machine gives the runs:
GPU memory, processors, shared memory. Check these. Then it trains every run
for 50 steps in `runs/trial`, and prints how long the rest will take for
several step budgets (also kept in `runs/time_estimate.txt`).

**This is the moment to choose another step budget**, if 400,000 steps per
run do not fit in the time you have. Stop the script (`Ctrl-c`) and start
it again with the number of steps:

```bash
bash scripts/run_comparison.sh runs 200000
```

The learning-rate schedule follows the budget, so the budget can change
only until the first run saves its first checkpoint (5,000 steps in). After
that, another budget needs a new output folder.

## While it runs

- **Progress:** `tail -f runs/proposal_comparison/suite.log` (a line every
  100 steps, with the time). If the first run trains much slower than the
  estimate, another workload may be sharing the GPU, or the volume is slow.
- **Low validation PSNR for most of a run is expected.** Validation uses
  one source view, but training uses two for the first three quarters of
  the steps. It rises once one-view steps begin.
- **After a crash, a restart or a stopped workload:** run the same command
  again (same output folder, same container image, volume at the same
  path). Finished work is kept, and training continues from its last
  checkpoint, saved every 5,000 steps.
- **To stop it:** `Ctrl-c` in tmux; with nohup,
  `pkill -f run_comparison.sh; pkill -f scene-recon`, and wait until
  `pgrep -f scene-recon` prints nothing before starting again.

## What not to do

- **Do not edit the configs once training has started.** Every training
  setting is fixed for a run that has trained; a run with changed settings
  refuses to continue. Only the settings in `CONTINUATION_SETTINGS`
  (`src/scene_recon/provenance.py`) may change: scoring, validation, speed
  measurement, logging, checkpoint interval and loading processes.
- **Do not start a second launch on the same folder.** The script refuses
  while another one, or a training process it started, still uses it.
- **Do not change the GPU share** or run other jobs on the GPU midway.
- **Do not delete `runs/proposal_comparison_first_100_test_scenes/`**: it
  is the report on the first 100 test cars, and step 2 does not run again
  while it exists. Delete it only after adding a model or a seed, so that
  step 2 trains and scores the new runs.

## Results

| File | Contents |
|---|---|
| `runs/proposal_comparison_first_100_test_scenes/summary.md` | The report on the first 100 test cars, after about 91 hours on a 4090 |
| `runs/proposal_comparison/summary.md` | The final report, on all 704 test cars |

Copy them off the machine before it is shut down, for example without the
checkpoints:

```bash
rsync -a --exclude checkpoints <user>@<server>:3d-scene-reconstruction/runs/ gpu-server-runs/
```

**If time runs out** during the scoring of all 704 cars, stop it: the
report on the first 100 cars is already complete. Scoring settings never
need training again, so fewer views per car (for example
`--set evaluation.num_target_views=10`, by hand as in the docs) score all
704 cars in a fraction of the time, though no longer by pixelNeRF's
benchmark.

## What matches pixelNeRF

Everything in the training and scoring follows pixelNeRF's code, the same
way for every model: 4 cars of 128 rays per step from all 50 views of each
car, two source views and rays from the box around the car for the first
three quarters of the steps, then one or two views and the whole image; the
learning rate 1e-4; 64 + 32 samples per ray, 16 of them around the depth the
first 64 found; its pixel centres; scores on every non-source test view,
rounded down to 8 bits, with random sample positions. The one difference: the
learning rate rises over the first 2,000 steps and then falls tenfold over
the budget (pixelNeRF keeps 1e-4 throughout).
