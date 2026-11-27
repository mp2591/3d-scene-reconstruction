# Starting the training

How to start the full comparison on a Linux GPU machine (for example the
Run:ai GH200), and what to keep in mind while it runs. Every detail is in
[`docs/running_on_a_gpu_server.md`](docs/running_on_a_gpu_server.md).

## What it trains

Three models, trained and scored at the same time on the one GPU, each with
pixelNeRF's recipe for ShapeNet-SRN Cars
(`configs/suites/proposal_comparison.yaml`). Their rendering speed is
measured at the end, one model at a time, with the GPU to itself:

| Run | Encoder | Decoder |
|---|---|---|
| `convnextv2_mlp` | ConvNeXt V2-T | pixelNeRF's MLP |
| `hiera_mlp` | Hiera-T | pixelNeRF's MLP |
| `hiera_nerf` | Hiera-T | HieraNeRF bridge and attention decoder |

Every run trains the same number of steps and is scored with 1 and 2
source views. The two baselines, `pixelnerf_resnet34` and `transnerf_unet`,
are commented out until their model files exist.

**Within 20 hours.** Started with `20h` (below), the script first times the
three runs together on this machine (about 15 minutes in). It then
chooses the step budget itself: the largest of 25,000, 50,000, ... up to
pixelNeRF's 400,000 whose report is estimated to be ready within 20 hours
of the start, with a tenth of the time to spare. That report scores 10
evenly spaced views of each of the first 100 test cars (1,000 views per
model and number of source views). pixelNeRF's full recipe does not fit:
400,000 steps and its benchmark, every view of all 704 test cars, take
about 368 hours on an RTX 4090. After the report, the script goes on to
that full benchmark (step 3); stop it when the GPU time is up.

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
      memory size"). Each run needs about 360 MB of `/dev/shm`, about 1.1 GB
      for the three at once; the default 64 MB is too little, and training
      refuses to start.
- [ ] **At least 30 processors** (each of the three runs uses 9), and
      generous main memory.
- [ ] **A fixed GPU share** (for example 0.99) that stays the same until
      the end, with nothing else scheduled on that GPU. The three runs need
      up to about 66 GB together. The speeds are only comparable when every
      run had the same GPU.

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
bash scripts/run_comparison.sh runs 20h
```

Leave tmux with `Ctrl-b` then `d`, come back with `tmux attach -t comparison`.
Without tmux:

```bash
nohup bash scripts/run_comparison.sh runs 20h > /dev/null 2>&1 &
```

`runs` is the output folder; use a new name if `runs` already holds an old
comparison. Everything is also written to `runs/run_comparison.log`. The 20
hours count from the first start in that folder (kept in `runs/started_at`),
also after a restart.

## 3. In the first 15 minutes

The script first checks the GPU and prints what the machine gives the runs:
GPU memory, processors, shared memory. Check these. Then it trains every run
for 50 steps in `runs/trial`, times the three runs together, prints how long
the rest takes for each step budget (also kept in `runs/time_estimate.txt`)
and the budget it chose: "Chosen for 20h: ... steps per run".

To choose the budget yourself instead, stop the script (`Ctrl-c`) and start
it again with the number of steps:

```bash
bash scripts/run_comparison.sh runs 200000
```

The learning-rate schedule follows the budget, so the budget can change
only until the first run saves its first checkpoint (5,000 steps in). After
that, another budget needs a new output folder.

## While it runs

- **Results so far:** `runs/proposal_comparison/summary.md` is rewritten
  every 30 minutes, with every run's status and its loss and validation
  PSNR curves so far (2 source views on 4 validation cars, every 5,000
  steps). TensorBoard shows the same live (see the docs).
- **Progress:** `tail -f runs/proposal_comparison/suite.log` (a line every
  100 steps per run, with the time and the run's name). If a run's time per 100 steps in `suite.log` is well above 100 times its
  "s per training step" in `time_estimate.txt` (where the three were timed
  together), something else is using the GPU and the hours will not hold.
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
| `runs/proposal_comparison_first_100_test_scenes/summary.md` | The report within the 20 hours: 10 views of each of the first 100 test cars |
| `runs/proposal_comparison/summary.md` | While training: the report so far. At the very end: pixelNeRF's full benchmark, every view of all 704 test cars |

Copy them off the machine before it is shut down, for example without the
checkpoints:

```bash
rsync -a --exclude checkpoints <user>@<server>:3d-scene-reconstruction/runs/ gpu-server-runs/
```

**When the GPU time is up** during step 3 (every view of all 704 cars),
stop it: the report on the first 100 cars is already complete.

## What matches pixelNeRF

Everything in the training and scoring follows pixelNeRF's code, the same
way for every model: 4 cars of 128 rays per step from all 50 views of each
car, two source views and rays from the box around the car for the first
three quarters of the steps, then one or two views and the whole image; the
learning rate 1e-4; 64 + 32 samples per ray, 16 of them around the depth the
first 64 found; its pixel centres; scores rounded down to 8 bits, with
random sample positions. The differences, all to fit in 20 hours except the
first:

- the learning rate rises over the first 2,000 steps and then falls tenfold
  over the budget (pixelNeRF keeps 1e-4 throughout);
- the budget is what fits, up to pixelNeRF's 400,000 steps, and the first
  phase lasts three quarters of it;
- the report within the 20 hours scores 10 of the about 250 non-source
  views of each of the first 100 test cars. pixelNeRF's full benchmark
  comes only in step 3.
