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
| `profile.json`        | Parameter counts and rendering speed (`profile`).              |

## Commands

| Command                   | Purpose                                                              |
|---------------------------|----------------------------------------------------------------------|
| `scene-recon train`       | Train a model. Add `--resume` to continue an interrupted run.        |
| `scene-recon evaluate`    | Compute PSNR, SSIM and LPIPS on the test split.                      |
| `scene-recon profile`     | Report parameter counts and full-frame rendering speed (FPS).        |
| `scene-recon check-model` | Verify that a model satisfies the pipeline's contract, in seconds.   |

Every command takes `--config <file>`. Single values can be overridden
without editing a file:

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

3. Write your model ([guide](docs/writing_a_model.md)), copy
   `configs/srn_cars_tiny_model.yaml`, and point it at your model.

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
docs/
  pipeline.md            How data flows through the pipeline, and its conventions.
  writing_a_model.md     The model contract and how to plug a network in.
  configuration.md       How config files are composed and validated.
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
tests/                   Unit tests and end-to-end tests.
```

## Tests

```bash
pytest
```

The suite runs on a CPU in under a minute. It includes a test that renders an
analytically known scene through the full ray casting and volume rendering
code and compares the result with the exact image, and end-to-end tests of
every command.
