# Sparse-View 3D Scene Reconstruction

A config-driven pipeline for training and evaluating generalizable neural
radiance fields: models that reconstruct a 3D scene from one to three images
in a single forward pass. It is the common ground for comparing encoder and
decoder architectures (pixelNeRF, TransNeRF, ConvNeXt V2, Hiera and a Mask
Unit attention decoder) on the ShapeNet-SRN Cars benchmark.

The pipeline owns everything that must be identical across experiments: data
loading, ray sampling, volume rendering, the loss, optimisation,
checkpointing, evaluation and profiling. The neural network is a plug-in: any
`nn.Module` that implements two methods can be trained and evaluated, and is
selected by name in a YAML file. A whole comparison is described by one suite
file: list the runs, start one command, and get every checkpoint, the test
metrics, a comparison table and the plots.

> **The repository contains no real architectures yet.** The only model
> included, `tiny_pixel_aligned_nerf`, is a few small layers that exist to
> test the pipeline. It must not be used for reported results.

**Contents:**
[Installation](#installation) ·
[Quick start](#quick-start) ·
[Running a whole comparison](#running-a-whole-comparison) ·
[Fair comparisons](#fair-comparisons-what-the-pipeline-guarantees) ·
[Writing a network](#writing-a-network-instructions-for-model-authors) ·
[Commands](#commands) ·
[Training on ShapeNet-SRN Cars](#training-on-shapenet-srn-cars) ·
[Repository layout](#repository-layout) ·
[Tests](#tests)

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

| File                  | Content                                                              |
|-----------------------|----------------------------------------------------------------------|
| `config.yaml`         | The complete config of the run, with all inheritance resolved.       |
| `checkpoints/last.pt` | Model, optimiser and scheduler state.                                |
| `tensorboard/`        | Loss curves, learning rates, validation metrics and renders.         |
| `evaluation.json`     | Test metrics, overall and per scene (`evaluate`).                    |
| `previews/`           | A few rendered test views, for side-by-side comparison (`evaluate`). |
| `profile.json`        | Parameter counts and rendering speed (`profile`).                    |

## Running a whole comparison

A suite file lists several runs that share most of their settings. One
command trains, evaluates and profiles them in order and writes a report
comparing them:

```bash
scene-recon run-suite --config configs/suites/smoke_suite.yaml
```

The report is at `runs/smoke_suite/summary.md`. It contains:

- what differs between the runs
- tables of training cost, speed and test metrics
- a scene-by-scene comparison of every run with the first one, with 95%
  confidence intervals (mean ± spread over seeds, if the suite lists several `seeds`)
- plots of quality against the number of input views, quality against
  rendering speed, and training curves
- the same test views rendered by every run, next to the ground truth

Starting the same command again resumes an interrupted suite and skips
everything that is already done.

[`docs/suites.md`](docs/suites.md) describes the suite file and the outputs.
[`configs/suites/proposal_template.yaml`](configs/suites/proposal_template.yaml)
lays out the comparison from the proposal.

## Fair comparisons: what the pipeline guarantees

A difference in results between two runs should come from the architecture,
not from the setup. The pipeline makes sure of that:

| What                     | Guarantee                                                                                         |
|--------------------------|---------------------------------------------------------------------------------------------------|
| **Input images**         | Every model receives the same images in the same form: RGB, values in [0, 1], at the dataset's native resolution (128×128 for ShapeNet-SRN), no augmentation. Normalisation for a pretrained backbone happens inside its encoder ([below](#3-normalise-the-input-inside-the-encoder)). |
| **Training data**        | Every model with the same seed sees exactly the same training data, step by step: the same scenes, loaded views, input views, supervised pixels and coarse sample positions along rays. (Fine-pass positions depend on each model's own coarse prediction, as hierarchical sampling intends.) It does not matter how the model initialises its weights or uses randomness, whether the run was interrupted and resumed, how many data-loading workers are used, or whether it runs on a CPU or a GPU. |
| **Measurement**          | Evaluation is deterministic: every model renders the same views of the same test scenes from the same input views and is scored by the same code. A suite refuses to start if its runs differ in any measurement setting (evaluation, validation, profiling, validation or test data). |
| **Everything else**      | May differ between runs on purpose (the model, an encoder's learning rate...). The suite report lists every differing setting, so it is visible what was varied and that everything else was held fixed. |
| **Provenance**           | Every result is tied to what produced it. A checkpoint is only continued, evaluated or profiled with the exact settings it was trained with; changing, say, the learning rate of a trained run is refused instead of silently mixed in. Checkpoints record a hash of the training data's content, and a run is not reused or resumed once that data has changed. Evaluations record the checkpoint and the test data by the hash of their content, profiles record the device, software and workload, and the report shows a result only while it still matches. |

Two limits remain, and the suite helps with both. Training is random, and
on a GPU not even repeatable bit for bit (some GPU operations add numbers in
a varying order), so the same model can end slightly differently. A suite
can repeat every run with several `seeds` and report the spread. And the
test set is a sample of scenes, so the report compares every run with the
first one scene by scene, with a 95% confidence interval, and calls a
difference better or worse only when the interval excludes zero. See
[`docs/suites.md`](docs/suites.md#several-seeds).

The mechanics are described in [`docs/pipeline.md`](docs/pipeline.md#the-training-data-schedule).

## Writing a network: instructions for model authors

You write the neural network; the pipeline does everything else. The pipeline
talks to exactly one object: your **top-level model**, an `nn.Module` that
contains your encoder, your decoder and anything else. What happens inside it
(the encoder's output format, how views are fused, how the decoder is
structured) is your design.

What the pipeline needs from you:

1. [Put your code in `src/scene_recon/models/`](#1-put-your-code-in-srcscene_reconmodels)
2. [Write the top-level model](#2-write-the-top-level-model)
3. [Normalise the input inside the encoder](#3-normalise-the-input-inside-the-encoder)
4. [Leave the data alone](#4-leave-the-data-alone)
5. [Pretrained weights and learning rates](#5-pretrained-weights-and-learning-rates)
6. [Optional: encoders and decoders chosen by config](#6-optional-encoders-and-decoders-chosen-by-config)
7. [Describe the model in a config file](#7-describe-the-model-in-a-config-file)
8. [Check it, then add it to a suite](#8-check-it-then-add-it-to-a-suite)

The complete reference is [`docs/writing_a_model.md`](docs/writing_a_model.md);
[`models/testing/tiny_model.py`](src/scene_recon/models/testing/tiny_model.py)
is a small, complete, working example.

### 1. Put your code in `src/scene_recon/models/`

Every Python file in that directory, and in sub-directories that have an
`__init__.py`, is imported automatically, which is what registers your
classes. A file anywhere else is not found.

```
src/scene_recon/models/
    pixelnerf/
        __init__.py
        encoders.py       # ResNet-34, ...
        decoders.py       # ResNet-block MLP, ...
        model.py          # the top-level model
```

### 2. Write the top-level model

Subclass `ReconstructionModel`, register it under a name, and implement two
methods:

```python
import torch

from scene_recon.models.base import ReconstructionModel
from scene_recon.registry import MODELS
from scene_recon.structures import FieldOutput


@MODELS.register("pixelnerf")                      # the name used in configs
class PixelNeRF(ReconstructionModel):
    def __init__(self, hidden_width: int = 512):   # every argument is a config key
        super().__init__()
        self.encoder = ...
        self.decoder = ...

    def encode_scene(self, source_views):
        """Called once per batch. Look at the input views; return anything."""
        images = source_views.images                # (B, V, 3, H, W), values in [0, 1]
        features = self.encoder(images.flatten(0, 1)).unflatten(0, images.shape[:2])
        return features, source_views.cameras, source_views.image_size

    def query_field(self, scene_encoding, points, view_directions, level):
        """Called per chunk of rays. Predict colour and density at 3D points."""
        # points: (B, R, S, 3) world coordinates, S samples per ray from near to far
        # view_directions: (B, R, 3); level: "coarse" or "fine"
        raw_rgb, raw_density = self.decoder(...)
        return FieldOutput(rgb=torch.sigmoid(raw_rgb), density=torch.relu(raw_density))
```

The rules:

| Rule | Why |
|---|---|
| `rgb` has shape (B, R, S, 3) with values in [0, 1]; `density` has shape (B, R, S) and is ≥ 0. Use e.g. `sigmoid` and `relu`/`softplus`. | The renderer integrates them as colour and density. |
| Work with **any** number of source views V (1 to 3 in the proposal); pool or attend over views instead of concatenating a fixed number. | V changes from step to step, and evaluation runs with 1, 2 and 3 views. |
| Create new tensors on the input's device: `torch.zeros(..., device=points.device)`. Never call `.cuda()`. | The same code runs on a CPU (tests) and on any GPU. |
| `level` tells the coarse pass from the fine pass. Use one network for both (ignore `level`) or one each, as pixelNeRF does. | Hierarchical sampling queries the model twice. |
| Every trainable parameter must affect the output. Delete unused layers (e.g. a backbone's classification head) or freeze them with `requires_grad = False`. | Unused parameters waste optimiser memory and usually signal a wiring mistake; `check-model` reports them. |

`geometry.py` provides the camera maths a pixel-aligned model needs:
`world_to_camera`, `rotate_directions_to_camera`, `project_to_pixels` and
`sample_features_at_pixels`. The tiny model shows them in sequence.

### 3. Normalise the input inside the encoder

Every model receives the same [0, 1] images. A pretrained backbone works as
intended only on inputs normalised the way it was pretrained (most use the
ImageNet mean and standard deviation), so each encoder applies its own
normalisation, first thing in `forward`, with the shared helper. Make it a
constructor argument, so it is visible in the config and in the suite report:

```python
from torch import nn

from scene_recon.models.normalization import InputNormalization
from scene_recon.registry import ENCODERS


@ENCODERS.register("hiera")
class HieraEncoder(nn.Module):
    def __init__(self, input_normalization: str = "imagenet"):
        super().__init__()
        self.normalize = InputNormalization(input_normalization)   # "imagenet", "minus_one_to_one" or "none"
        self.backbone = ...

    def forward(self, images):              # (N, 3, H, W), values in [0, 1]
        return self.backbone(self.normalize(images))
```

Use `"none"` for an encoder trained from scratch. To support other weights,
add their statistics to `NORMALIZATIONS` in
[`models/normalization.py`](src/scene_recon/models/normalization.py).

### 4. Leave the data alone

Everything about the data is decided by the pipeline, identically for every
model. Inside your modules:

| Don't | Why |
|---|---|
| Augment images (random crops, flips, resizing, colour jitter) | Each model would see different data. Crops, flips and resizes would also no longer match the camera parameters. If augmentation is wanted, it gets added to the pipeline for every model. |
| Pick, drop or reorder input views | Which views a model sees is part of the shared training data. |
| Call `torch.manual_seed(...)` or other seeding functions | The pipeline seeds from `experiment.seed` and restores the random state when a run is resumed. Reseeding breaks both. |
| Read data, cameras or files during `forward` | All inputs arrive as arguments. Loading weights once in `__init__` is fine (next section). |

Dropout and stochastic depth are fine: they are part of the model.

### 5. Pretrained weights and learning rates

Load pretrained weights in `__init__`, from a path passed in through the
config, never a hard-coded one. To give a pretrained encoder its own learning
rate and warmup, split the parameters into named groups:

```python
from scene_recon.models.base import ParameterGroup

    def parameter_groups(self):
        return [
            ParameterGroup("encoder", list(self.encoder.parameters())),
            ParameterGroup("decoder", list(self.decoder.parameters())),
        ]
```

```yaml
optimization:
  learning_rate: 1.0e-4                    # the decoder
  group_overrides:
    encoder: {learning_rate: 1.0e-5, warmup_steps: 2000}
```

Every trainable parameter must be in exactly one group. For layer-wise
learning-rate decay, use one group per stage (`encoder.stage0`,
`encoder.stage1`, ...) with a decreasing `lr_scale`. The override for
`encoder` applies to all of them.

### 6. Optional: encoders and decoders chosen by config

To try many encoder–decoder combinations without writing a top-level class
for each, register encoders and decoders, and let one top-level model build
them by name:

```python
from scene_recon.registry import DECODERS, ENCODERS


@MODELS.register("pixel_aligned_nerf")
class PixelAlignedNeRF(ReconstructionModel):
    def __init__(self, encoder: dict, decoder: dict):
        super().__init__()
        self.encoder = ENCODERS.build(encoder)
        self.decoder = DECODERS.build(decoder, num_features=self.encoder.num_features)
```

The top-level code is then shared by every combination, so **every encoder
must produce what every decoder expects**. Agree on a convention, for example
"every encoder has a `num_features` attribute and returns a list of feature
maps", or adapt inside the top-level model.

### 7. Describe the model in a config file

One file per model in `configs/models/`. `name` selects the registered class;
every other key is passed to its constructor.

```yaml
# configs/models/hiera_maskunit.yaml
model:
  name: pixel_aligned_nerf
  encoder:
    name: hiera
    input_normalization: imagenet
    weights: /data/weights/hiera_base_mae.pth
  decoder:
    name: mask_unit_attention
    num_heads: 8
```

### 8. Check it, then add it to a suite

```bash
scene-recon check-model --config configs/my_experiment.yaml
```

This builds the model and runs it on random scenes at the dataset's image
size, querying real samples along real camera rays with your configured
numbers of samples per ray and input views. It reports the first broken rule
in plain words: wrong shapes, colours outside [0, 1], negative density or
NaNs, and, if the model will be trained, parameters missing from the
parameter groups or receiving no gradient or a non-finite one (remove or
freeze parameters you don't use). It takes seconds. A model used only for
inference (evaluation and profiling) can be checked without the training
checks: `--stages evaluate profile`. A model without trainable parameters
(fully frozen, or analytic) cannot be trained and needs no checkpoint: it is
evaluated as built, seeded with `experiment.seed`, so randomly initialised
frozen layers are the same every time. A suite runs the same check for the
stages it runs, plus everything else each run will need, before it starts.

Then add the model as a run to a suite file:

```yaml
runs:
  hiera_maskunit:
    base: ../models/hiera_maskunit.yaml
```

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

3. Write your models ([instructions above](#writing-a-network-instructions-for-model-authors)),
   describe each in `configs/models/`, and list them as runs in a copy of
   `configs/suites/proposal_template.yaml`.

## Repository layout

```
configs/                 Experiment configs (YAML).
  base.yaml              Every pipeline setting, with the values from the proposal.
  data/                  One file per dataset.
  models/                One file per model.
  suites/                Suite files: several runs and a comparison report.
docs/
  pipeline.md            How data flows through the pipeline, and its conventions.
  writing_a_model.md     The model contract in full.
  configuration.md       How config files are composed and validated.
  suites.md              Running and comparing many experiments from one file.
src/scene_recon/
  cli.py                 Command-line entry point.
  config.py              Config loading and validation.
  registry.py            Name -> class registries behind the configs.
  structures.py          Data containers and the shape conventions.
  geometry.py            Camera maths: rays, projection, feature lookup.
  data/                  Datasets (ShapeNet-SRN, synthetic spheres), the training
                         data schedule, and batch loading.
  rendering/             Ray selection, sampling along rays, volume rendering.
  models/                The model contract and input normalisation; your models go here.
    testing/             The tiny testing model.
  losses.py  metrics.py  optim.py  checkpoints.py
  training.py            The training loop.
  evaluation.py          Novel-view evaluation.
  profiling.py           Parameter counts and rendering speed.
  stages.py              Train, evaluate and profile one run (used by every command).
  run_files.py           Where each output of a run is stored.
  suite.py               Loading a suite file and running its experiments in order.
  fairness.py            What must match across runs, and what differs.
  provenance.py          Which checkpoint and settings produced a result, and whether it is still valid.
  comparison.py          Spread over seeds, and paired per-scene differences between runs.
  environment.py         The software, hardware and source revision recorded with every result.
  report.py  plots.py    The suite report: tables and figures.
tests/                   Unit tests and end-to-end tests.
```

## Tests

```bash
pytest
```

GitHub runs the same tests on every push and pull request, on Python 3.10
and 3.13 ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).

The tests run on a CPU in about a minute. They include:

- a test that renders an analytically known scene through the full ray
  casting and volume rendering code and compares the result with the exact image
- tests that two different models are trained on identical data, and that an
  interrupted and resumed run (with dropout) ends with exactly the same
  weights as an uninterrupted one
- tests that results are never attributed to the wrong settings, checkpoint
  or device, and that damaged results files are redone
- end-to-end tests of every command and of suites
