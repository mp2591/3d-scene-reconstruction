# Writing a model

The pipeline never looks inside a model. It needs exactly two operations,
defined by `ReconstructionModel` in
[`src/scene_recon/models/base.py`](../src/scene_recon/models/base.py):

```python
class ReconstructionModel(nn.Module):
    def encode_scene(self, source_views: SceneViews) -> Any: ...
    def query_field(self, scene_encoding, points, view_directions, level) -> FieldOutput: ...
```

How the work is divided between an encoder and a decoder, how views are
fused, and whether attention runs across views or along rays is all up to the
model.

## The contract

### `encode_scene(source_views)`

Called once per batch of scenes, before any rendering.

| Input                                    | Shape           | Notes                           |
|------------------------------------------|-----------------|---------------------------------|
| `source_views.images`                    | (B, V, 3, H, W) | RGB in [0, 1]                   |
| `source_views.cameras.camera_to_world`   | (B, V, 4, 4)    | OpenCV convention               |
| `source_views.cameras.intrinsics`        | (B, V, 3, 3)    | In pixels                       |

`V`, the number of source views, changes from call to call (1 to 3 in the
proposal). A model must not assume a fixed `V`.

Return whatever the model needs later, typically feature maps together with
the source cameras. The pipeline hands the value back to `query_field`
unchanged.

### `query_field(scene_encoding, points, view_directions, level)`

Called many times per scene, once per chunk of rays.

| Input             | Shape        | Notes                                                    |
|-------------------|--------------|----------------------------------------------------------|
| `points`          | (B, R, S, 3) | World coordinates. `S` samples per ray, near to far.     |
| `view_directions` | (B, R, 3)    | Unit vectors in world coordinates, one per ray.          |
| `level`           | `str`        | `"coarse"` or `"fine"`: which rendering pass is asking.  |

Points arrive grouped by ray and ordered by depth, so a decoder can attend
across the samples of a ray, as the TransNeRF ray decoder does. `S` differs
between the coarse and the fine pass.

Return a `FieldOutput`:

| Output    | Shape        | Requirement                                   |
|-----------|--------------|-----------------------------------------------|
| `rgb`     | (B, R, S, 3) | In [0, 1], for example after a sigmoid.       |
| `density` | (B, R, S)    | Non-negative, for example after a ReLU.       |

A model may use separate networks for the two levels (as pixelNeRF does) or
ignore `level` and share one.

### `parameter_groups()` (optional)

By default all parameters are trained with the same learning rate. Override
this method to split them into named groups:

```python
def parameter_groups(self) -> list[ParameterGroup]:
    return [
        ParameterGroup("encoder", list(self.encoder.parameters())),
        ParameterGroup("decoder", list(self.decoder.parameters())),
    ]
```

Group names are what the config refers to:

```yaml
optimization:
  learning_rate: 1.0e-4            # the decoder uses this
  group_overrides:
    encoder: {learning_rate: 1.0e-5, warmup_steps: 2000}
```

For layer-wise learning-rate decay, return one group per encoder stage, named
`encoder.stage0`, `encoder.stage1`, and so on, each with its own `lr_scale`.
An override for `encoder` applies to all of them, and `lr_scale` multiplies
the result.

## Step by step

1. **Create a file** in `src/scene_recon/models/`, for example
   `pixelnerf.py`. Every file in that directory is imported automatically.

2. **Write and register the model.**

   ```python
   from scene_recon.models.base import ReconstructionModel
   from scene_recon.registry import MODELS
   from scene_recon.structures import FieldOutput, SceneViews


   @MODELS.register("pixelnerf")
   class PixelNeRF(ReconstructionModel):
       def __init__(self, hidden_width: int = 512) -> None:
           super().__init__()
           ...

       def encode_scene(self, source_views: SceneViews):
           ...

       def query_field(self, scene_encoding, points, view_directions, level) -> FieldOutput:
           ...
   ```

3. **Write its config**, `configs/models/pixelnerf.yaml`. Every key other than
   `name` is passed to `__init__`.

   ```yaml
   model:
     name: pixelnerf
     hidden_width: 512
   ```

4. **Write an experiment config** that combines it with a dataset.

   ```yaml
   base:
     - base.yaml
     - data/srn_cars.yaml
     - models/pixelnerf.yaml

   experiment:
     run_name: pixelnerf_srn_cars
   ```

5. **Check the contract** before spending GPU time. This runs the model on
   random inputs and reports the first violation in plain words.

   ```bash
   scene-recon check-model --config configs/pixelnerf_srn_cars.yaml
   ```

   The random test images have the size of the dataset's images, so encoders
   with a fixed input size (such as position embeddings) are checked at the
   size they will see. The queried points are real samples along real camera
   rays, ordered from near to far, with the configured numbers of samples per
   ray, so a decoder that attends along rays is checked on the input it will
   actually get, with the numbers of input views your config uses. If the
   model will be trained, the check also runs a backward pass with each
   configured number of training views (`training.num_source_views`): every
   trainable parameter must receive a finite gradient from at least one of
   them. Delete or freeze (`requires_grad = False`) parameters your model
   does not use, such as a pretrained backbone's classification head. A
   model without trainable parameters (fully frozen, or analytic with no
   parameters at all) can be evaluated and profiled, as built and without a
   checkpoint; check it with `--stages evaluate profile`.

6. **Train**, on its own or as one run of a suite ([suites.md](suites.md)).

   ```bash
   scene-recon train --config configs/pixelnerf_srn_cars.yaml
   ```

## Input normalisation

Every model receives the same images: RGB, values in [0, 1]. Keeping the
input identical is what makes the comparison fair. A pretrained backbone,
however, works as intended only on inputs normalised the way it was
pretrained, and backbones differ. So each encoder normalises its own input,
first thing in `forward`, with the shared helper in
[`models/normalization.py`](../src/scene_recon/models/normalization.py):

```python
from scene_recon.models.normalization import InputNormalization

class ConvNeXtV2Encoder(nn.Module):
    def __init__(self, input_normalization: str = "imagenet") -> None:
        super().__init__()
        self.normalize = InputNormalization(input_normalization)
        ...

    def forward(self, images: Tensor) -> Tensor:   # (N, 3, H, W) in [0, 1]
        return self.stages(self.normalize(images))
```

Available: `"imagenet"` (most pretrained backbones), `"minus_one_to_one"`
and `"none"` (for encoders trained from scratch). Check which one your
weights were pretrained with, and add an entry to `NORMALIZATIONS` if it is
missing. As a constructor argument the choice appears in the model's config,
and in the suite report when runs differ in it. The helper's statistics are
not stored in checkpoints, so the config always decides.

## What a model must not do

The pipeline decides everything about the data, identically for every model
(see [pipeline.md](pipeline.md#the-training-data-schedule)). So inside a model:

- **No augmentation.** Random crops, flips, resizing or colour jitter would
  give each model different data, and geometric changes would no longer match
  the camera parameters. Augmentation, if wanted, belongs in the pipeline.
- **No choosing of views.** Use all source views you are given.
- **No seeding.** Do not call `torch.manual_seed` or similar; the pipeline
  seeds from `experiment.seed` and restores the random state on resume.
- **No reading of data during `forward`.** Load weights once, in `__init__`,
  from a path given in the config.

Dropout, stochastic depth and other randomness that is part of the model are
fine.

## Encoders and decoders inside the top-level model

The pipeline sees only the top-level model. What happens inside it (which
encoder and decoder it contains, what the encoder hands to the decoder, how
features are fused across views) is entirely up to whoever writes it.

### Optional: choosing the encoder and decoder from config

The proposal pairs several encoders with several decoders. Instead of
writing one top-level class per combination, a top-level model can build its
parts by name from the `ENCODERS` and `DECODERS` registries:

```python
@ENCODERS.register("resnet34")
class ResNet34Encoder(nn.Module): ...

@DECODERS.register("resnet_mlp")
class ResNetMLPDecoder(nn.Module): ...

@MODELS.register("pixel_aligned_nerf")
class PixelAlignedNeRF(ReconstructionModel):
    def __init__(self, encoder: dict, decoder: dict) -> None:
        super().__init__()
        self.encoder = ENCODERS.build(encoder)
        self.decoder = DECODERS.build(decoder, num_features=self.encoder.num_features)
```

Swapping the backbone is then a config change:

```yaml
model:
  name: pixel_aligned_nerf
  encoder: {name: hiera}          # was: resnet34
  decoder: {name: resnet_mlp}
```

[`models/testing/tiny_model.py`](../src/scene_recon/models/testing/tiny_model.py)
is a complete, small example of this pattern.

One thing to plan for if you use it: the top-level model's code is shared by
every combination, so every encoder must produce what every decoder expects.
For example, if the shared code samples one feature map per view, an encoder
returning several multi-scale maps needs an adapter, or a shared convention
such as "every encoder returns a list of maps". `scene-recon check-model`
catches a mismatch in seconds.

Encoder and decoder files must live under `src/scene_recon/models/`, like the
top-level model, so that their registration runs.

## Geometry helpers

Pixel-aligned models need to find, for each 3D point, the matching location
in each source image. [`geometry.py`](../src/scene_recon/geometry.py) provides
tested functions for this, so all models share one set of conventions:

| Function                      | What it does                                                             |
|-------------------------------|--------------------------------------------------------------------------|
| `world_to_camera`             | Express world points in each source camera's frame.                      |
| `rotate_directions_to_camera` | The same for directions (no translation).                                |
| `project_to_pixels`           | Project camera-space points to pixel coordinates.                        |
| `sample_features_at_pixels`   | Bilinearly read feature maps of any resolution at those pixel positions. |

The testing model's `query_field` uses all four in sequence.
