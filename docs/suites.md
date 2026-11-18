# Running a suite of experiments

A suite file describes every run of a comparison in one place. One command
trains, evaluates and profiles each run in order, then writes a report that
compares them:

```bash
scene-recon run-suite --config configs/suites/smoke_suite.yaml
```

The smoke suite finishes in a few minutes on a CPU and is the quickest way to
see what a suite produces. [`configs/suites/proposal_template.yaml`](../configs/suites/proposal_template.yaml)
is the starting point for the comparison in the proposal.

## The suite file

```yaml
suite_name: proposal_comparison        # results go to runs/proposal_comparison/
stages: [train, evaluate, profile]     # any subset; always executed in this order

common:                                # merged into every run
  base: [../base.yaml, ../data/srn_cars.yaml]

runs:                                  # executed top to bottom
  pixelnerf_resnet34:
    base: ../models/pixelnerf_resnet34.yaml

  hiera_mlp:
    model:
      name: pixel_aligned_nerf
      encoder: {name: hiera}
      decoder: {name: resnet_mlp}
    optimization:
      group_overrides:
        encoder: {learning_rate: 1.0e-5, warmup_steps: 2000}
```

The file has exactly four keys:

| Key          | Meaning                                                                         |
|--------------|---------------------------------------------------------------------------------|
| `suite_name` | Name of the suite and of its output directory.                                  |
| `stages`     | Which of `train`, `evaluate` and `profile` to execute for each run.             |
| `common`     | A config fragment shared by every run.                                          |
| `runs`       | One config fragment per run. The key is the run's name.                         |

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
[writing_a_model.md](writing_a_model.md)). A run that already has a
checkpoint must request exactly the training settings it was trained with.
All problems of all runs are reported together, and nothing is trained until
they are fixed. A typo in the last run costs seconds, not the days spent
training the runs before it.

The runs must also be **measured identically**: the `evaluation`,
`validation` and `profiling` sections and the validation and test data must
be the same for every run, or the suite does not start. Set them in
`common`. Runs measured differently could not be compared.

**2. The runs are executed one after another**, each through the requested
stages. Work that is already done is skipped:

| Stage    | Skipped when                                                                   | Otherwise                       |
|----------|--------------------------------------------------------------------------------|---------------------------------|
| train    | the checkpoint has reached `training.num_steps`                                | resumed if a checkpoint exists, started if not |
| evaluate | `evaluation.json` was made from the current checkpoint file (compared by the hash of its content, not its name or step) with the current evaluation, renderer and test-data settings | evaluated again |
| profile  | `profile.json` was made with the current model, renderer and profiling settings, on the same kind of device with the same PyTorch version | profiled again |

A damaged or half-written results file counts as missing and is redone.

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
reason. If the report itself cannot be written, the error is logged and the
remaining runs continue.

## Outputs

```
runs/<suite_name>/
  suite.log                          everything the suite printed, with timestamps
  summary.md                         the report: tables and figures
  summary.csv                        the same numbers, one row per run
  plots/
    metrics_vs_source_views.png      each metric against the number of input views
    quality_vs_speed.png             PSNR against rendering speed
    training_curves.png              training loss and validation PSNR, by step and by training time
    renders_<n>_source_views.png     the same test views rendered by every run, next to the truth
  <run_name>/                        one directory per run, as for a single run
```

Each run directory has the same layout as a run started with the single-run
commands (see [pipeline.md](pipeline.md)), including its own `config.yaml`. So
every single-run command works on a suite run too, for example to re-evaluate
one run with a different checkpoint:

```bash
scene-recon evaluate --config runs/proposal_comparison/hiera_mlp/config.yaml --checkpoint my.pt
```

The report starts with **what differs between runs**: a table of every
setting whose value is not the same for all runs (the model's settings, an
encoder's learning rate, a renderer ablation, ...). Everything not in the
table is identical, and the report states when all runs were also trained on
identical data, step for step (same seed, training data and batch settings;
see [pipeline.md](pipeline.md#the-training-data-schedule)).

`summary.csv` has one row per run. Quality columns are named
`<metric>_<n>v`: `psnr_1v` is the PSNR with one input view. Load it with
pandas or a spreadsheet for analysis the report does not cover.

The render comparison shows the first non-source view of the first
`evaluation.num_preview_scenes` test scenes. These renders are saved by every
evaluation, so the comparison is available even with `save_images: false`.

## Trying a suite quickly

`--set` changes every run, which makes a short trial of the whole suite easy
before committing GPU days to it:

```bash
scene-recon run-suite --config configs/suites/proposal_template.yaml --set training.num_steps=200 training.validate_every=100 evaluation.num_scenes=5
```

The trial writes into the same directory as the real suite would. Delete
`runs/<suite_name>/` afterwards, or the real run will resume from the trial's
checkpoints.

## Comparing runs during training

The report is the comparison at the end of each run. To watch all runs of a
suite while they train, point TensorBoard at the suite directory:

```bash
tensorboard --logdir runs/proposal_comparison
```
