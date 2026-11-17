# Sparse-View 3D Scene Reconstruction

A config-driven pipeline for training and evaluating generalizable neural
radiance fields: models that reconstruct a 3D scene from one to three images
in a single forward pass. It is the common ground for comparing encoder and
decoder architectures (pixelNeRF, TransNeRF, ConvNeXt V2, Hiera and a Mask
Unit attention decoder) on the ShapeNet-SRN Cars benchmark.

The pipeline owns everything that should be identical across experiments:
data loading, ray sampling, volume rendering, the loss, optimisation,
checkpointing, evaluation and profiling. The neural network is a plug-in. Any
`nn.Module` that implements two methods can be trained and evaluated, and is
selected by name in a YAML file.

A whole comparison is described by one suite file: list the runs, start one
command, and get every checkpoint, the test metrics, a comparison table and
the plots.

> **The repository contains no real architectures yet.** The only model
> included, `tiny_pixel_aligned_nerf`, is a few small layers that exist to
> test the pipeline. It must not be used for reported results.

## Installation

Python 3.10 or newer is required.

```bash
pip install -e ".[dev]"
```

LPIPS is an optional extra because it downloads pretrained weights on first
use. Install it before running a full evaluation:

```bash
pip install -e ".[lpips]"
```

## Quick start

The smoke test trains the testing model on procedurally generated scenes. It
needs no dataset and finishes in about a minute on a CPU.

```bash
scene-recon train --config configs/smoke_test.yaml
```

```bash
scene-recon evaluate --config configs/smoke_test.yaml
```

```bash
tensorboard --logdir runs
```

Outputs are written to `runs/<run_name>/`:

| File                  | Content                                                        |
|-----------------------|----------------------------------------------------------------|
| `config.yaml`         | The complete config of the run, with all inheritance resolved. |
| `checkpoints/last.pt` | Model, optimiser and scheduler state.                          |
| `tensorboard/`        | Loss curves, learning rates, validation metrics and renders.   |
| `evaluation.json`     | Test metrics, overall and per scene (`evaluate`).              |
| `previews/`           | A few rendered test views, for side-by-side comparison (`evaluate`). |
| `profile.json`        | Parameter counts and rendering speed (`profile`).              |

## Running a whole comparison

A suite file lists several runs that share most of their settings. One
command trains, evaluates and profiles them in order and writes a report
comparing them:

```bash
scene-recon run-suite --config configs/suites/smoke_suite.yaml
```

The report is at `runs/smoke_suite/summary.md`: tables of training cost,
speed and test metrics, with plots of quality against the number of input
views, quality against rendering speed, training curves, and the same views
rendered by every run. Starting the same command again resumes an interrupted
suite and skips everything that is already done.

[`docs/suites.md`](docs/suites.md) describes the suite file and the outputs.
[`configs/suites/proposal_template.yaml`](configs/suites/proposal_template.yaml)
lays out the comparison from the proposal.

## Commands

| Command                   | Purpose                                                              |
|---------------------------|----------------------------------------------------------------------|
| `scene-recon train`       | Train a model. Add `--resume` to continue an interrupted run.        |
| `scene-recon evaluate`    | Compute PSNR, SSIM and LPIPS on the test split.                      |
| `scene-recon profile`     | Report parameter counts and full-frame rendering speed (FPS).        |
| `scene-recon check-model` | Verify that a model satisfies the pipeline's contract, in seconds.   |
| `scene-recon run-suite`   | Run every experiment of a suite file and write a comparison report.  |

Every command takes `--config <file>`: an experiment config, or a suite file
for `run-suite`. Single values can be overridden without editing a file
(for `run-suite`, in every run):

```bash
scene-recon train --config configs/smoke_test.yaml --set training.batch_size=8 experiment.run_name=bigger_batch
```

## Training on ShapeNet-SRN Cars

1. Extract the dataset and set the three `split_dir` paths in
   [`configs/data/srn_cars.yaml`](configs/data/srn_cars.yaml).
2. Check that the pipeline reads your copy of the data by training the testing
   model for a short while:

   ```bash
   scene-recon train --config configs/srn_cars_tiny_model.yaml --set training.num_steps=200 training.validate_every=100
   ```

3. Write your models ([guide](docs/writing_a_model.md)), describe each in
   `configs/models/`, and list them as runs in a copy of
   `configs/suites/proposal_template.yaml`.

## Adding your own model

In short: create a file in `src/scene_recon/models/`, subclass
`ReconstructionModel`, implement `encode_scene` and `query_field`, and register
the class under a name.

```python
@MODELS.register("my_model")
class MyModel(ReconstructionModel):
    def encode_scene(self, source_views): ...
    def query_field(self, scene_encoding, points, view_directions, level): ...
```

```yaml
model:
  name: my_model
```

[`docs/writing_a_model.md`](docs/writing_a_model.md) explains the contract,
the geometry helpers available to models, and how to make encoders and
decoders interchangeable from config.

## Repository layout

```
configs/                 Experiment configs (YAML).
  base.yaml              Every pipeline setting, with the values from the proposal.
  data/                  One file per dataset.
  models/                One file per model.
  suites/                Suite files: several runs and a comparison report.
docs/
  pipeline.md            How data flows through the pipeline, and its conventions.
  writing_a_model.md     The model contract and how to plug a network in.
  configuration.md       How config files are composed and validated.
  suites.md              Running and comparing many experiments from one file.
src/scene_recon/
  cli.py                 Command-line entry point.
  config.py              Config loading and validation.
  registry.py            Name -> class registries behind the configs.
  structures.py          Data containers and the shape conventions.
  geometry.py            Camera maths: rays, projection, feature lookup.
  data/                  Datasets (ShapeNet-SRN, synthetic spheres) and batching.
  rendering/             Ray selection, sampling along rays, volume rendering.
  models/                The model contract; your models go here.
    testing/             The tiny testing model.
  losses.py  metrics.py  optim.py  checkpoints.py
  training.py            The training loop.
  evaluation.py          Novel-view evaluation.
  profiling.py           Parameter counts and rendering speed.
  stages.py              Train, evaluate and profile one run (used by every command).
  run_files.py           Where each output of a run is stored.
  suite.py               Loading a suite file and running its experiments in order.
  report.py  plots.py    The suite report: tables and figures.
tests/                   Unit tests and end-to-end tests.
```

## Tests

```bash
pytest
```

The tests run on a CPU in under a minute. It includes a test that renders an
analytically known scene through the full ray casting and volume rendering
code and compares the result with the exact image, and end-to-end tests of
every command.
