# How the pipeline works

## One training step

```
SceneDataset ──► batch of scenes, a few random views each        data/
                   │
                   ├─► pick 1-3 of the views as source views     training.py
                   │
                   └─► sample random pixels from all views,      rendering/rays.py
                       cast a ray through each
                             │
model.encode_scene(source views)                                 your model
                             │
for the coarse pass, then the fine pass:                         rendering/renderer.py
    choose depths along each ray                                 rendering/sampling.py
    model.query_field(points) ──► colour and density             your model
    integrate along each ray  ──► one colour per ray             rendering/volume.py
                             │
loss(rendered colours, true pixel colours)                       losses.py
                             │
optimiser step, learning-rate schedule                           optim.py
```

The trainer ([`training.py`](../src/scene_recon/training.py)) is a plain
PyTorch loop that can be read from top to bottom. It counts steps rather than
epochs, logs to TensorBoard, validates on a few held-out scenes at regular
intervals, and saves a checkpoint from which a run can be resumed.

## The training data schedule

For a fair comparison every model must be trained on the same data. The
`TrainingSchedule` ([`data/schedule.py`](../src/scene_recon/data/schedule.py))
makes every random choice about the training data a function of two numbers
only, `experiment.seed` and the step number:

| Choice                                  | Decided by                          |
|-----------------------------------------|-------------------------------------|
| Which scenes form the batch             | `schedule.scene_plan(step)`         |
| Which views of each scene are loaded    | `schedule.scene_plan(step)`         |
| Which loaded views are the model's input | `schedule.source_view_indices(step)` |
| Which pixels are supervised             | `schedule.generator("rays", step)`  |
| Where along each ray samples are placed | `schedule.generator("sample_depths", step)` |

Each choice has its own random generator, seeded from the seed, the choice's
name and the step. PyTorch's global random generator is never used for data:
it belongs to the model (weight initialisation, dropout). As a result:

- **Every model sees the same data at every step**, however many random
  numbers it draws itself. Only the fine-pass sample positions differ, because
  they depend on each model's own coarse prediction.
- **A resumed run continues exactly where it stopped.** The data of step
  *n* does not depend on what happened before step *n*. Checkpoints also
  store the global random states (PyTorch, Python, NumPy), and the data
  loader has its own generator, so the model's own randomness (dropout,
  stochastic depth) continues exactly too. On a CPU, an interrupted and
  resumed run ends with the same weights, bit for bit, as an uninterrupted
  one; a test checks this for a model with dropout, with and without
  data-loading worker processes.
- **A run keeps its settings.** A checkpoint can only be resumed with the
  settings it was trained with, apart from e.g. the step budget (see
  [`provenance.py`](../src/scene_recon/provenance.py)), so a run's weights
  always correspond to one config.
- **Workers and devices make no difference.** The choices are made in the
  main process, on the CPU, before any data is loaded.

Scenes are visited in epochs: each epoch is a fresh random order of the
training scenes, and every scene appears at most once per epoch.

Images reach the model exactly as the dataset loads them: RGB, values in
[0, 1], native resolution, no augmentation. Normalisation for a pretrained
backbone is the encoder's job; see
[writing_a_model.md](writing_a_model.md#input-normalisation).

## Evaluation

For each test scene, the evaluator
([`evaluation.py`](../src/scene_recon/evaluation.py)) shows the model a fixed
set of source views, renders every other view in full, and compares each
rendered image with the ground truth. Metrics are averaged over the views of
a scene, then over scenes.

`evaluation.source_view_sets` lists the protocols to run. Each entry is one
set of input views, so `[[64], [64, 104]]` evaluates the model with one input
view and again with two.

Every evaluation also saves the first rendered view of the first
`evaluation.num_preview_scenes` scenes under `previews/`. All models evaluated
with the same source views render the same views, so these images can be
compared side by side; the suite report does exactly that.

## The three cost figures of the proposal

| Figure             | Where it comes from                                                           |
|--------------------|-------------------------------------------------------------------------------|
| Training GPU hours | Recorded during training; stored in the checkpoint and in `evaluation.json`.  |
| Parameter count    | `scene-recon profile`, in total and per parameter group.                      |
| Rendering FPS      | `scene-recon profile`: time to render one full frame of an encoded scene, averaged over `profiling.num_frames` frames. |

Training time counts optimisation steps and data loading, and excludes
validation and checkpointing. It accumulates correctly across resumed runs.

## Conventions

These hold everywhere in the code base. They are also stated at the top of
[`structures.py`](../src/scene_recon/structures.py) and
[`geometry.py`](../src/scene_recon/geometry.py).

**Shape letters.** `B` scenes per batch, `N` views per scene (`V` when they
are source views), `R` rays per scene, `S` samples per ray, `H` and `W` image
height and width.

**Cameras.** OpenCV convention: in the camera frame, +x is right, +y is down
and +z points into the scene. Poses are camera-to-world matrices. This is how
ShapeNet-SRN stores its poses, so they are used without conversion.

**Pixels.** Coordinates are ordered (x, y). The centre of the top-left pixel
is at (0.5, 0.5). The original SRN and pixelNeRF code reads the SRN
intrinsics with pixel centres at integers instead, half a pixel apart; the
SRN loader's `pixel_centers` setting chooses between the two (see
[`data/srn.py`](../src/scene_recon/data/srn.py)). Which one matches the
renders is still to be established on the real data.

**Rays and depth.** Ray directions have unit length, so a depth is a distance
along the ray, and `near` and `far` are distances from the camera.

**Images.** Float RGB in [0, 1], channels first: (3, H, W).

**Background.** Rays that hit nothing show `renderer.background_color`, white
for ShapeNet-SRN.

## Rendering details

**Coarse pass.** `num_coarse_samples` depths per ray, one from each of that
many equal bins between `near` and `far`. During training the position inside
each bin is random; during evaluation it is the bin centre, which makes
renders deterministic.

**Fine pass.** `num_fine_samples` additional depths are drawn where the coarse
pass placed rendering weight (the hierarchical sampling of NeRF). The model is
then queried at the union of coarse and fine depths with `level="fine"`. Set
`num_fine_samples: 0` to disable the pass.

**Loss.** Mean squared error between rendered and true colours, summed over
the coarse and the fine render.

**Training rays.** Rays are drawn uniformly from all pixels of all loaded
views of a scene, source views included, as in pixelNeRF.

## Adding a dataset

Subclass `SceneDataset` ([`data/base.py`](../src/scene_recon/data/base.py)),
implement `__len__`, `num_views` and `load_scene`, register the class with
`@DATASETS.register("name")`, and import the module in
[`data/__init__.py`](../src/scene_recon/data/__init__.py). A dataset only
loads views; the trainer and evaluator decide which ones are inputs and which
are targets.

Give all scenes of a split the same image size. The encoders accept only
sizes their strides divide, and the checks before a run read the size of
each split's first scene only.

Also override `fingerprint()` to identify the data's content: it is
recorded with every checkpoint and evaluation, an evaluation made on other
test data is redone, and a run is not resumed on other training data. The
default only records the class and the number of scenes. Hash the
*content* of everything `load_scene` reads, as the SRN loader does (images,
poses and intrinsics), so that data replaced in place, even by files of the
same size, is recognised. The SRN loader keeps the hash for the rest of the
process only while every file's size, modification time, change time and
inode are unchanged, and hashes again otherwise.

Checkpoints from before content hashing record an older SRN fingerprint
(`files_sha256`: names and sizes only). It cannot confirm that the files
still hold the same content, so such a checkpoint is not resumed or reused
for training; the error says so. It can still be evaluated and profiled.
To train further, start a new run.

## Adding a loss or a metric

Both follow the same registry pattern; see
[`losses.py`](../src/scene_recon/losses.py) and
[`metrics.py`](../src/scene_recon/metrics.py) for the expected signatures.
