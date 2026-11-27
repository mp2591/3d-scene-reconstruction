# Running a suite of experiments

A suite file describes every run of a comparison in one place. One command
trains, evaluates and profiles each run in order, then writes a report that
compares them:

```bash
scene-recon run-suite --config configs/suites/smoke_suite.yaml
```

The smoke suite finishes in a few minutes on a CPU and is the quickest way to
see what a suite produces. [`configs/suites/proposal_comparison.yaml`](../configs/suites/proposal_comparison.yaml)
is the comparison in the proposal, ready to run; its comments give the time
it takes. Every run in it trains with pixelNeRF's recipe, untuned
([running_on_a_gpu_server.md](running_on_a_gpu_server.md), "The training
recipe"); `scripts/run_comparison.sh` runs it from start to finish.

## The suite file

```yaml
suite_name: proposal_comparison        # results go to runs/proposal_comparison/
stages: [train, evaluate, profile]     # any subset; always executed in this order

common:                                # merged into every run
  base: [../base.yaml, ../data/srn_cars.yaml]

runs:                                  # executed top to bottom
  pixelnerf_resnet34:
    base: ../models/pixelnerf_resnet34.yaml

  convnextv2_mlp:
    model:
      name: pixel_aligned_nerf
      encoder: {name: convnextv2}
      decoder: {name: pixelnerf_mlp}
```

The file has four required keys and three optional keys:

| Key          | Meaning                                                                         |
|--------------|---------------------------------------------------------------------------------|
| `suite_name` | Name of the suite and of its output directory.                                  |
| `stages`     | Which of `train`, `evaluate` and `profile` to execute for each run.             |
| `common`     | A config fragment shared by every run.                                          |
| `runs`       | One config fragment per run. The key is the run's name.                         |
| `seeds`      | Optional. Repeat every run once per seed, e.g. `[0, 1, 2]`; see [Several seeds](#several-seeds). |
| `parallel_runs` | Optional. How many runs train and are scored at the same time, e.g. `3`. Without it, one after another. |
| `report_every_minutes` | Optional. While runs train in parallel, also rewrite the report this often, e.g. `30`, with every run's curves so far. Without it, the report is rewritten after each run. |

`common` and each run are ordinary config fragments, written exactly like the
config files described in [configuration.md](configuration.md). They can use
`base` (paths relative to the suite file) and set any section. A run's
complete config is built in this order, later entries winning:

1. `common`, including the files named in its `base`
2. the run's own entry, including the files named in its `base`
3. `--set` overrides given on the command line

The run's key becomes its `experiment.run_name`. All runs write into
`<output_dir>/<suite_name>/`, where `output_dir` is `common`'s
`experiment.output_dir` (`runs` in `base.yaml`).

A model can be described in a file of its own (`base: ../models/x.yaml`) or
written out in the run (`model: {...}`). Files are convenient when the same
model appears in several suites.

## What happens when a suite runs

**1. Every run is checked before anything starts.** For each run the config
is validated, and everything the requested stages will need is built: the
datasets (with their view indices checked), the loss, the metrics (including
optional packages such as LPIPS), the optimiser and its parameter groups, and
the model, which is checked against the pipeline's contract (see
[writing_a_model.md](writing_a_model.md)), only with the checks its stages
need (a model used only for evaluation and profiling may be fully frozen,
or have no parameters at all). Every evaluation and validation protocol must
leave at least one view to render in every scene it uses; profiling only
needs that for the one view it renders (the first test scene, with the first
evaluation source-view set). Evaluation without training needs a checkpoint,
unless the model has no trainable parameters: such a model cannot be
trained, so it is evaluated as built, with the weights a training run
would start from (seeded with `experiment.seed`, so a model with random
frozen weights is the same model every time). A run that already has a
checkpoint must request exactly the training settings it was trained with
and, when the suite trains, the training data must still be the data the
checkpoint recorded, even for a run that has finished and would be skipped. All
problems of all runs are reported together, and nothing is trained until
they are fixed. A typo in the last run costs seconds, not the days spent
training the runs before it.

The runs must also be **measured identically**: the `evaluation`,
`validation` and `profiling` sections, the validation and test data, and
`renderer.ray_chunk_size` (rendering speed depends on it) must be the same
for every run, or the suite does not start. Set them in `common`. Runs
measured differently could not be compared. Profiling also stops with an
error if rendering filled the GPU's memory (less than
`profiling.min_free_gpu_memory` of it free when read, after each frame is
queued and once all have run): some drivers (on Windows) then
move memory to the computer's main memory instead of failing, and the
measured speed would be several times too low. Each `profile.json` records
the peak GPU memory of rendering.

**2. The runs are executed**, each through the requested stages: one after
another, or with `parallel_runs`, that many at the same time, each in a
process of its own on the same GPU. Training and scores come out the same
either way. Rendering speed does not, so with `parallel_runs` it is
measured after all training and scoring, one run at a time, with the GPU to
itself. The training hours are then measured while the runs share the GPU:
they say how long each run took next to the others, not how fast its model
trains alone. Work that is already done is skipped:

| Stage    | Skipped when                                                                   | Otherwise                       |
|----------|--------------------------------------------------------------------------------|---------------------------------|
| train    | the checkpoint has reached `training.num_steps`                                | resumed if a checkpoint exists, started if not |
| evaluate | `evaluation.json` was made from the current checkpoint file (compared by the hash of its content, not its name or step) with the current evaluation, renderer and test-data settings | evaluated again |
| profile  | `profile.json` was made with the current model, renderer and profiling settings, on the same kind of device with the same PyTorch and `timm` versions | profiled again |

A damaged or half-written results file counts as missing and is redone. If
the training progress next to a checkpoint (`checkpoints/last.json`) is
damaged, it is read from the checkpoint itself instead. If the version of
`timm` cannot be read, a saved profile cannot be checked against it, so the
run is profiled again and the report leaves its speed out, saying why.

As a result:

- **An interrupted suite is resumed by running the same command again.**
- **Adding a run** to the suite file and running it again only trains the new run.
- **Raising `training.num_steps`** continues every run from its checkpoint and
  evaluates it again.
- **Changing evaluation settings** (say, adding LPIPS) evaluates every run
  again without retraining.
- **Changing a trained run's training settings** (its model, seed, training
  data, loss, rendering, optimisation or batch settings) is refused during the
  check, with a list of what changed: the checkpoint was not trained that way.
  Give the run a new name, or delete its directory to train it from scratch.
  Only settings that do not change the trained weights may change for an
  existing run: the step budget, device, workers, logging and checkpoint
  intervals, and everything about measurement. The full list is
  `CONTINUATION_SETTINGS` in
  [`provenance.py`](../src/scene_recon/provenance.py).

**3. If a run fails**, for example by running out of GPU memory, the error is
logged, the suite moves on to the next run, and the command exits with an
error listing the failed runs once all runs are done. Fix the problem and run
the suite again; finished runs are skipped.

**4. The report is rewritten after every run**, so partial results can be
read while later runs are still training. It shows a run's results only while
they still match the run's current checkpoint and settings; results that no
longer match are left out of every table and figure and listed with the
reason. If the report cannot be written after a run, the error is logged
and the remaining runs continue; if the final report still cannot be
written, the command fails.

Run names must not be the names of the suite's own outputs (`suite.log`,
`summary.md`, `summary.csv`, `paired_comparison.csv`, `plots`, in any
upper/lower case), because those sit next to the run directories.

## Outputs

```
runs/<suite_name>/
  suite.log                          everything the suite printed, with timestamps
  summary.md                         the report: tables, comparisons and figures
  summary.csv                        the same numbers, one row per run and seed
  paired_comparison.csv              per-scene differences to the reference run, with confidence intervals
  plots/
    metrics_vs_source_views.png      each metric against the number of input views
    quality_vs_speed.png             PSNR against rendering speed
    training_curves.png              training loss and validation PSNR, by step and by training time
    renders_<n>_source_views.png     the same test views rendered by every run, next to the truth
  <run_name>/                        one directory per run, as for a single run
                                     (with seeds: <run_name>/seed_<seed>/ per seed)
```

Each run directory has the same layout as a run started with the single-run
commands (see [pipeline.md](pipeline.md)), including its own `config.yaml`. So
every single-run command works on a suite run too, for example to re-evaluate
one run with a different checkpoint:

```bash
scene-recon evaluate --config runs/proposal_comparison/hiera_mlp/seed_0/config.yaml --checkpoint my.pt
```

The report starts with **what differs between runs**: a table of every
setting whose value is not the same for all runs (the model's settings, an
encoder's learning rate, a renderer ablation, ...). Everything not in the
table is identical, and the report states when all runs were also trained on
identical data, step for step (same seed, training data and batch settings;
see [pipeline.md](pipeline.md#the-training-data-schedule)).

The report then **compares every run with the reference**, the first run of
the suite file, so list your baseline first. Every model is evaluated on the
same test scenes, so the comparison is made scene by scene: for each scene,
the run's score minus the reference's, averaged over the scenes, with a 95%
confidence interval (Student's t over scenes). A run is called *better* or
*worse* only when the interval excludes zero; otherwise the report says
*no clear difference*. The same numbers are in `paired_comparison.csv`. If
the reference has no results yet, the report says so rather than comparing
against another run.

A result only counts for the comparison once the run's checkpoint has
exactly the requested number of training steps. If, say, a suite extends
training from 4 000 to 6 000 steps and one run's extension fails, that run's
evaluation of its 4 000-step checkpoint is still a valid measurement of that
checkpoint, but not of the requested experiment; the same holds for a run
trained past the requested steps. The report shows it in the quality tables
marked with \*, with its actual step count, and leaves it out of best
values, quality figures and paired comparisons, saying why. It counts again
once its steps match. The training curves include it; a mean over seeds
covers the steps every seed has logged. When a
suite does not request training (`stages: [evaluate]`), the existing
checkpoints are what is being compared, so their results always count. A run
whose evaluation finished but whose profiling failed keeps its comparison.

Whether the runs were trained on identical data is read from what their
checkpoints recorded, not only from the config; if the checkpoints record
different training data, the report says so prominently and shows each
run's data.

The same holds for pretrained weights. Every checkpoint records the weights
its run started from: where they came from (a file path or a `timm` name)
and a SHA-256 hash of the weights as loaded. The table shows them for each
run, and if runs that name the same file started from different weights (the
file was replaced, or a teammate downloaded another version), the report
says so prominently. If the seeds of one run started from different weights,
each seed is listed.

A saved evaluation stays valid when the software is updated, so that the
report can also be made on another machine. Every evaluation saves what it
was made with: the Python version, the CUDA version PyTorch was built with,
the type and model of the device (floating-point results can differ slightly
between GPU models; on the CPU, the number of threads too) and every
installed package with its version, so that a package a new model or metric
uses is covered without anyone naming it.

The report compares these records whole: two evaluations count as made with
the same software when all of these are identical, and only then. What the
record leaves out, it cannot see: two builds of a package that carry the
same version number, the GPU driver, the operating system, and the source
revision (a change that alters results raises the compatibility version
instead). If two runs' records differ, it warns and lists only what differs.
If a record cannot be compared, it says why: *not recorded* (made before
evaluations recorded their software), *device model unknown* (only NVIDIA
GPUs, and CPUs whose model the operating system reports, are named),
*package list unreadable* (the installed packages could not all be read;
the evaluation's log names the package's folder when one is to blame, and
otherwise says to check the folders on Python's search path) or *damaged*. Such a record is never
taken for the same software. The paired comparisons of those runs then give
no plain verdict: *scored with different software* or *unknown* (`software`
in `paired_comparison.csv`: `same`, `different` or `unknown`). `summary.csv`
has a short hash of each run's record. The report cannot tell which package
a score depends on, so it may flag an update that does not matter for it;
the list lets you judge. Evaluate every run again with one installation
(`scene-recon evaluate`) for an exact comparison.

The report trusts the files this program wrote. It checks that they can be
read, and treats a damaged one (cut short, not JSON, or with a key given
twice) like a missing one, so the work is redone. A file edited by hand can
still claim anything, its scores included, so the report does not try to
catch hand edits.

`summary.csv` has one row per run and seed. Quality columns are named
`<metric>_<n>v`: `psnr_1v` is the PSNR with one input view. Load it with
pandas or a spreadsheet for analysis the report does not cover.

The render comparison shows the first non-source view of the first
`evaluation.num_preview_scenes` test scenes. These renders are saved by every
evaluation, so the comparison is available even with `save_images: false`.

## Several seeds

Training is random: the same architecture trained with another seed ends
somewhat differently. To see how large that effect is, list seeds:

```yaml
suite_name: proposal_comparison
seeds: [0, 1, 2]
```

Every run is then executed once per seed, in `<run_name>/seed_<seed>/`, with
`experiment.seed` set to that seed (a seed set in a run's own config is
replaced). For each seed, all runs are still trained on identical data, so
runs stay comparable seed by seed. The cost grows with the number of seeds.

The report then shows each metric as mean ± standard deviation over the
seeds, plots error bars, and averages each run's training curves over its
seeds. The paired comparison uses only the seeds that both runs have results
for, so a seed that has finished for one run but not yet for the other
cannot shift the result; the compared seeds are listed, and a comparison
that does not yet cover every seed is marked *partial*. A difference is convincing when its confidence
interval excludes zero *and* it is larger than the spread between seeds.

## Trying a suite quickly

`--set` changes every run, which makes a short trial of the whole suite easy
before committing GPU days to it:

```bash
scene-recon run-suite --config configs/suites/proposal_comparison.yaml --set training.num_steps=200 training.validate_every=100 evaluation.num_scenes=5 experiment.output_dir=runs/trial
```

Without `experiment.output_dir`, the trial writes into the same directory as
the real suite would. Delete `runs/<suite_name>/` afterwards, or the real run
will resume from the trial's checkpoints.

Checkpoints are large (several hundred MB, rewritten every
`training.checkpoint_every` steps). If the repository sits in a synced
folder such as OneDrive, point `experiment.output_dir` outside it for real
runs, or every checkpoint is uploaded.

## Comparing runs during training

The report is the comparison at the end of each run. To watch all runs of a
suite while they train, point TensorBoard at the suite directory:

```bash
tensorboard --logdir runs/proposal_comparison
```
